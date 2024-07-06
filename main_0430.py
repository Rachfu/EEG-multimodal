import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from functools import partial
import pickle
import torch.distributed as dist
from functools import partial
from torch.optim import Adam
import warnings
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder,BertConfig
from sklearn.metrics import f1_score
import os
from opacus import PrivacyEngine
# sss
def set_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

set_seed(980616)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings('ignore')

class MultiModalDataset_ti(Dataset):
    '''
    treat eeg as txt, action as img. ti means txt + img
    '''
    def __init__(self,eeg_df_path,action_path,eeg_path):
        self.eeg_df = pd.read_csv(eeg_df_path)
        self.label = self.eeg_df['label']
        with open(action_path, 'rb') as f:
            self.train_clip_feature = pickle.load(f)
        with open(eeg_path, 'rb') as f:
            self.train_text_embedding = pickle.load(f)

    def __len__(self):
        return len(self.eeg_df)

    def __getitem__(self,idx):
        video_feature = torch.tensor(self.train_clip_feature[idx])
        video_feature = video_feature.unsqueeze(0) # torch.Size([1, 1,512])
        mask = torch.tensor([1])
        input_ids = torch.tensor(self.train_text_embedding[idx]['input_ids']) # torch.Size([1, 512])
        attention_mask = torch.tensor(self.train_text_embedding[idx]['attention_mask'])

        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])

        return video_feature,mask,input_ids,attention_mask,label

def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

def DP_guarantee(feature,EPSILON,dp_mode=None):
    if dp_mode == 'feature_all_lap':
        feature_min = torch.min(feature, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature, dim=-1, keepdims=True)[0]
        feature = (feature - feature_min) / (feature_max - feature_min)
        lap_sigma=1/EPSILON
        m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
        noise = m.sample([feature.shape[0]])
        feature += noise.view(-1, 1)
    return feature


class ConcatModel(nn.Module):
    def __init__(self,args,dp_mode=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        # self.fc1 = nn.Linear(3*768, 3*768)
        # self.fc2 = nn.Linear(3*768, 768)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.dp_mode = dp_mode
        self.EPSILON = args.EPSILON

    def forward(self, frame_input, vedio_mask,title_input, text_mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_embedding = self.visual_encoder(frame_input)
        bert_semantics_embedding, bert_feature = self.bert(input_ids= title_input, attention_mask=text_mask,return_dict=False) #768
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                   tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
        cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1) #768
        
        img_feature = vision_embedding.squeeze(1)
        feature_concat = torch.cat((bert_feature, img_feature,cross_attn_result_text),dim=1)  # new add
        feature_concat = DP_guarantee(feature_concat,self.EPSILON,dp_mode=None).to(device)
        # feature_dnn = self.fc2(torch.relu(self.fc1(feature_concat)))
        feature_dnn = self.fc_layers(feature_concat)
        prediction = self.classifier(feature_dnn)
        # prediction = self.fc_layers(feature_concat)
        return prediction

def mkpath(path):
    return path+'/whole_record.txt', path+'/best_record.txt',path + '/best_f1.pickle'


def train(args,train_dataset,val_dataset,model,pretrain=False,load_stat=False):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if pretrain: # for pretrain
        os.makedirs(args.path+'/pretrain', exist_ok=True)
        whole_record_path,best_record_path,save_model_path = mkpath(args.path+'/pretrain')
    else: # for finetune
        if load_stat: # if ever pretrained
            load_model_path = args.path+'/pretrain/best_f1.pickle'
            model.load_state_dict(torch.load(load_model_path), strict=False)
        os.makedirs(args.path, exist_ok=True)
        whole_record_path,best_record_path,save_model_path = mkpath(args.path)

    if pretrain:
        trainable_layers = [model.bert.encoder.layer[-1],model.fc_layers,model.classifier]
        # trainable_layers = [model.classifier]
        for p in model.parameters():
            p.requires_grad = False

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
        DELTA = 1 / len(train_dataloader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=DELTA,
            target_epsilon=args.EPSILON, 
            epochs=args.epochs,
            max_grad_norm=args.MAX_GRAD_NORM,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # train!
    f1_score_best = 0.5
    for epoch in range(args.epochs):
        epoch_acc_train = 0
        epoch_loss_train = 0
        epoch_acc_val = 0
        epoch_loss_val = 0
        sample_size_train = 0
        sample_size_val = 0

        model.train()
        for frame_input, vedio_mask,title_input, text_mask, label in tqdm(train_dataloader):
            sample_size_train+=1
            model.train()
            optimizer.zero_grad()
            frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
            prediction = model(frame_input, vedio_mask,title_input, text_mask)
            loss, accuracy, _, _ = cal_loss(prediction,label)  
            epoch_loss_train += loss.item()
            epoch_acc_train += accuracy.item()
            loss.backward()
            optimizer.step()

        prediction_all = []
        label_all = []
        
        # eval!
        with torch.no_grad():
            for frame_input, vedio_mask,title_input, text_mask, label in tqdm(val_dataloader):
                sample_size_val +=1
                frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
                prediction = model(frame_input, vedio_mask,title_input, text_mask)
                loss, accuracy, pred_label_id, label_id = cal_loss(prediction,label)  # 3.26修改加f_1 score
                # loss, accuracy, _, _ = cal_loss(prediction, label)
                prediction_all.extend(pred_label_id.cpu().numpy())
                label_all.extend(label_id.cpu().numpy())
                epoch_loss_val += loss.item()
                epoch_acc_val += accuracy.item()

        f1_score_epoch = f1_score(prediction_all,label_all)

        record = f'''Epochs: {epoch + 1}
        | Train Loss: {epoch_loss_train/sample_size_train: .3f}
        | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
        | Val Loss: {epoch_loss_val/sample_size_val: .3f}
        | Val Accuracy: {epoch_acc_val/sample_size_val: .3f}
        | f_1 Score: {f1_score_epoch: .3f}\n'''
        print(record)
        with open(whole_record_path, "a") as file:
            file.write(record)

        if f1_score_epoch > f1_score_best:
            torch.save(model.state_dict(), save_model_path)
            f1_best_record = record
            # batch_size_record = str(batch_size)
            # with open(record_path, "w") as file:
            #     file.write(record+batch_size_record)
            f1_score_best = f1_score_epoch
            with open(best_record_path, "w") as file:
                file.write(f1_best_record)

def main(args):
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    # pretrain with SGD privacy!
    model = ConcatModel(args)
    train(args,train_dataset,val_dataset,model,pretrain=True,load_stat=False)
    # finetune with feature privacy!
    model = ConcatModel(args,dp_mode = 'feature_all_lap')
    train(args,train_dataset,val_dataset,model,pretrain=False,load_stat=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--epsilon",
    default=3,
    type=float,
    help="DP epsilon",
)
    main(args)
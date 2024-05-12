import torch
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings('ignore')

# * clip_path -> action_path
# * text_path -> eeg_path

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

def loss_function(prediction, label, model, weight_decay,epsilon):
    label = label.squeeze(dim=1)
    # cross_entropy_loss = F.cross_entropy(outputs, label.squeeze())
    cross_entropy_loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]

    tmp = (1-model.w)*np.exp(epsilon)+model.w
    loss_w,_ = torch.max(tmp, dim=0)
    total_loss = (1-weight_decay)*cross_entropy_loss + weight_decay *loss_w
    return total_loss,accuracy,pred_label_id, label

def gumbel_dropout(input,w,tau=0.1,hard=True):
    w_tensor = w.unsqueeze(1)
    one_minus_w_tensor = 1 - w_tensor
    w_tensor = torch.cat([w_tensor, one_minus_w_tensor], dim=1)
    gumbel_w = F.gumbel_softmax(w_tensor,tau,hard)
    mask = gumbel_w[:, 1]
    return input*mask/(1.0-w)

class GumbelSoftmaxDropout(nn.Module):
    def __init__(self, tau):
        super(GumbelSoftmaxDropout, self).__init__()
        self.tau = tau
    def forward(self, input,w):
        if self.training:
            res = gumbel_dropout(input,w,self.tau,hard=False)
        else:
            res = gumbel_dropout(input,w,self.tau,hard=True)
        return res
    
def Lap_noise(input,epsilon):
    pooled_output_min = torch.min(input, dim=-1, keepdims=True)[0]
    pooled_output_max = torch.max(input, dim=-1, keepdims=True)[0]
    pooled_output = (input - pooled_output_min) / (pooled_output_max - pooled_output_min)
    lap_sigma=1/epsilon
    m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = m.sample([pooled_output.shape[0]]).to(device)
    pooled_output += noise.view(-1, 1)
    return pooled_output

class ConcatModel(nn.Module):
    def __init__(self,tau, epsilon):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)

        self.classifier = nn.Linear(768*3, 2) # without fc_layers

        self.w = torch.nn.parameter.Parameter(data=torch.rand(768))
        self.dropout = GumbelSoftmaxDropout(tau)
        self.fc1 = nn.Linear(768*3, 768*3)
        self.fc2 = nn.Linear(768*3, 768)
        self.classifier = nn.Linear(768, 2) # without fc_layers
        self.epsilon = epsilon


    def forward(self, frame_input, vedio_mask,title_input, text_mask):
        vision_embedding = self.visual_encoder(frame_input)
        bert_semantics_embedding, bert_feature = self.bert(input_ids= title_input, attention_mask=text_mask,return_dict=False) #768
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                   tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
        cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1) #768
        
        img_feature = vision_embedding.squeeze(1)
        feature_concat = torch.cat((bert_feature, img_feature,cross_attn_result_text),dim=1)  # new add
        x = F.relu(self.fc1(feature_concat))
        x = self.fc2(x)
        res = self.dropout(x,self.w)
        res_lap = Lap_noise(res,self.epsilon)
        prediction = self.classifier(res_lap)
        return prediction

def pretrain():
    batch_size = 8
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    tau = 0.1
    epsilon = 0.1
    weight_decay = 0.5
    model = ConcatModel(tau, epsilon)
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    learning_rate = 1e-6
    epochs = 50
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    path = 'model_dict/PriGumbel/pretrain/'

    os.makedirs(path, exist_ok=True)
    
    # save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    # record_path = 'model_dict/ConcatModel/record.txt'
    # load_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    whole_record_path = path+'whole_record.txt'
    best_record_path = path+'best_record.txt'
    save_model_path = path+'best_f1.pickle'
    # model.load_state_dict(torch.load(load_model_path), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #! 在cpu上能跑
    # device = torch.device("cpu")
    # model = model.to(device)

    # model = SingleStream().to(device)
    model = model.to(device)  # 不并行试试
    # model = torch.nn.parallel.DataParallel(model.to(device))

    total_acc_val = 0
    total_loss_val = 0
    f1_score_best = 0.5

    # training
    for epoch in range(epochs):
        epoch_acc_train = 0
        epoch_loss_train = 0
        epoch_acc_val = 0
        epoch_loss_val = 0
        sample_size_train = 0
        sample_size_val = 0

        model.train()
        for frame_input, vedio_mask,title_input, text_mask, label in tqdm(train_dataloader):
            sample_size_train+=1
            optimizer.zero_grad()
            frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
            prediction = model(frame_input, vedio_mask,title_input, text_mask)
            # loss, accuracy, _, _ = cal_loss(prediction,label)  
            loss, accuracy, pred_label_id, label_id = loss_function(prediction, label, model, weight_decay,epsilon)
            epoch_loss_train += loss.item()
            epoch_acc_train += accuracy.item()
            # loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()

        prediction_all = []
        label_all = []
        # model.eval()
        model.eval()
        with torch.no_grad():
            for frame_input, vedio_mask,title_input, text_mask, label in tqdm(val_dataloader):
                sample_size_val +=1
                frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
                prediction = model(frame_input, vedio_mask,title_input, text_mask)
                loss, accuracy, pred_label_id, label_id = loss_function(prediction, label, model, weight_decay,epsilon)
                # loss, accuracy, pred_label_id, label_id = cal_loss(prediction,label)  # 3.26修改加f_1 score
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

def main():
    # settings
    # batch_size = 8
    # model = SingleStream()
    # epochs = 20
    # train_dataset = SingleStreamDataset('feature/train_EEG.csv')
    # val_dataset = SingleStreamDataset('feature/test_EEG.csv')
    # os.makedirs('model_dict/ConcatModel', exist_ok=True)
    # save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    # record_path = 'model_dict/ConcatModel/record.txt'
    batch_size = 8
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ConcatModel()
    trainable_layers = [model.bert.encoder.layer[-1],model.fc_layers,model.classifier]
    # trainable_layers = [model.classifier]
    for p in model.parameters():
        p.requires_grad = False

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True

    # optimizer = Adam(model.parameters(), lr=learning_rate)
    learning_rate = 1e-6
    epochs = 50
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    EPSILON = 7.5
    DELTA = 1 / len(train_dataloader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees
    MAX_GRAD_NORM = 0.1
    privacy_engine = PrivacyEngine()
    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_delta=DELTA,
        target_epsilon=EPSILON, 
        epochs=epochs,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    os.makedirs('model_dict/PriConcat/fineturn', exist_ok=True)
    
    # save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    # record_path = 'model_dict/ConcatModel/record.txt'
    load_model_path = 'model_dict/PriConcat/pretrain/best_f1.pickle'
    whole_record_path = 'model_dict/PriConcat/fineturn/whole_record.txt'
    best_record_path = 'model_dict/PriConcat/fineturn/best_record.txt'
    save_model_path = 'model_dict/PriConcat/fineturn/best_f1.pickle'
    model.load_state_dict(torch.load(load_model_path), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = SingleStream().to(device)
    model = model.to(device)  # 不并行能跑
    # model = torch.nn.parallel.DataParallel(model.to(device))

    total_acc_val = 0
    total_loss_val = 0
    f1_score_best = 0.5

    # training
    for epoch in range(epochs):
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
        # model.eval()
        eps = privacy_engine.get_epsilon(DELTA)

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
        | f_1 Score: {f1_score_epoch: .3f} 
        | epsilon: {eps: .3f}\n'''
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

def main2():
    '''
    Adding noise with Lap to all features
    '''
    # settings
    # batch_size = 8
    # model = SingleStream()
    # epochs = 20
    # train_dataset = SingleStreamDataset('feature/train_EEG.csv')
    # val_dataset = SingleStreamDataset('feature/test_EEG.csv')
    # os.makedirs('model_dict/ConcatModel', exist_ok=True)
    # save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    # record_path = 'model_dict/ConcatModel/record.txt'
    batch_size = 8
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ConcatModel(dp_mode = 'feature_all_lap')

    # optimizer = Adam(model.parameters(), lr=learning_rate)
    learning_rate = 1e-6
    epochs = 50
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    EPSILON = 7.5
    DELTA = 1 / len(train_dataloader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees


    
    os.makedirs('model_dict/seedPriConcat/fineturn3', exist_ok=True)
    
    # save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    # record_path = 'model_dict/ConcatModel/record.txt'
    load_model_path = 'model_dict/PriConcat/pretrain/best_f1.pickle'
    whole_record_path = 'model_dict/seedPriConcat/fineturn3/whole_record.txt'
    best_record_path = 'model_dict/seedPriConcat/fineturn3/best_record.txt'
    save_model_path = 'model_dict/seedPriConcat/fineturn3/best_f1.pickle'
    
    # model.load_state_dict(torch.load(load_model_path), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = SingleStream().to(device)
    model = model.to(device)  # 不并行能跑
    # model = torch.nn.parallel.DataParallel(model.to(device))

    total_acc_val = 0
    total_loss_val = 0
    f1_score_best = 0.5

    # training
    for epoch in range(epochs):
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
        # model.eval()

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
def test():
    model = ConcatModel(dp_mode='non-pri')
    # print(model.bert.encoder.layer[-1])
    load_model_path = 'model_dict/PriConcat/pretrain/best_f1.pickle'
    model.load_state_dict(torch.load(load_model_path), strict=False)
    weights = model.state_dict()
    for name, weight in weights.items():
        print(name, weight)



    # for name, module in model.named_children():
    #     print(name)


if __name__ == '__main__':
    pretrain()
    # main()
    # main2()
    # test()
    


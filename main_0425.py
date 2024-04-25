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
import clip


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
    
class MultiModal_ti(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.classifier = nn.Linear(768, 2)
        # self.classifier = nn.Linear(3*768, 2)

    def forward(self, frame_input, vedio_mask,title_input, text_mask):
        vision_embedding = self.visual_encoder(frame_input)
        bert_semantics_embedding = self.bert(input_ids=title_input, attention_mask=text_mask)['last_hidden_state']
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                   tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
        cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1)
        # feature_concat = torch.cat((vision_embedding.squeeze(1), bert_semantics_embedding[:, 0, :],cross_attn_result_text),dim=1)  # new add
        prediction = self.classifier(bert_semantics_embedding[:, 0, :])
  
        return prediction
    
def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


def main1():
    print('This is the result of method DoubleStream_ti')
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    
    # print('This is the result of method DoubleStream_tt')
    # train_dataset = MultiModalDataset_tt('feature/train_EEG.csv','feature/action/train_bert.pickle','feature/EEG/train_bert.pickle')
    # val_dataset = MultiModalDataset_tt('feature/test_EEG.csv','feature/action/test_bert.pickle','feature/EEG/test_bert.pickle')

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    step = 0
    max_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MultiModal_ti().to(device)
    model = MultiModal_ti()
    # model = MultiModal_tt()
    # model = SingleModal_tt()
    model = torch.nn.parallel.DataParallel(model.to(device))

    learning_rate = 5e-5
    optimizer = Adam(model.parameters(), lr=learning_rate)
    f1_score_best = 0.5

    for epoch in range(max_epochs):
        epoch_acc_train = 0
        epoch_loss_train = 0
        epoch_acc_val = 0
        epoch_loss_val = 0
        sample_size_train = 0
        sample_size_val = 0 

        for frame_input, vedio_mask,title_input, text_mask, label in tqdm(train_dataloader):
            sample_size_train+=1
            model.train()
            frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
            prediction = model(frame_input, vedio_mask,title_input, text_mask)
            loss, accuracy, _, _ = cal_loss(prediction,label)  
            epoch_loss_train += loss.item()
            epoch_acc_train += accuracy.item()
            loss.backward()
            optimizer.step()

        # validation
        prediction_all = []
        label_all = []

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

            # prediction_all, label_all = np.array(prediction_all),np.array(label_all)
            # print(prediction_all)
            # prediction_all = prediction_all.flatten()
            # label_all = label_all.flatten()
            # print(prediction_all)


            f1_score_epoch = f1_score(prediction_all,label_all)

            print(
                f'''Epochs: {epoch + 1}
            | Train Loss: {epoch_loss_train/sample_size_train: .3f}
            | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
            | Val Loss: {epoch_loss_val/sample_size_val: .3f}
            | Val Accuracy: {epoch_acc_val/sample_size_val: .3f}
            | f_1 Score: {f1_score_epoch: .3f}''')

            if f1_score_epoch > f1_score_best:
                torch.save(model.state_dict(), 'model_dict/DoubleStream/best_f1.pickle')
                f1_score_best = f1_score_epoch

class SingleStreamDataset(Dataset):
    def __init__(self,df_path):
        self.df = pd.read_csv(df_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.token = []
        self.label = []
        for i in range(len(self.df)):
            self.token.append(self.tokenizer(self.df.loc[i][0],padding='max_length',truncation=True,max_length=512))
            self.label.append(self.df.loc[i][1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        input_ids = torch.LongTensor(self.token[idx]['input_ids'])
        attention_mask = torch.LongTensor(self.token[idx]['attention_mask'])
        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return input_ids,attention_mask,label

class SingleStream(nn.Module):
    def __init__(self):
        super(SingleStream, self).__init__()
        # 加载bert
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_output_size = 768
        num_label = 2

        self.classifier = nn.Linear(bert_output_size, num_label)

    def forward(self,input_ids,attention_mask):
        _, pooled_output = self.bert(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)
        prediction = self.classifier(pooled_output)
        return prediction
    
class ConcatModelDataset(Dataset):
    def __init__(self,eeg_df_path,img_tensor_path):
        self.eeg_df = pd.read_csv(eeg_df_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_model, _ = clip.load("ViT-B/16",device="cpu")
        with open(img_tensor_path, 'rb') as f:
            self.img_tensor = pickle.load(f)

        self.token = []
        self.label = []
        self.img_input = []
        for i in range(len(self.eeg_df)):
            self.token.append(self.tokenizer(self.eeg_df.loc[i][0],padding='max_length',truncation=True,max_length=512))
            self.label.append(self.eeg_df.loc[i][1])
            self.img_input.append(self.img_model.encode_image(self.img_tensor[i].unsqueeze(0)))

    def __len__(self):
        return len(self.eeg_df)

    def __getitem__(self,idx):
        text_input = torch.LongTensor(self.token[idx]['input_ids'])
        text_attention = torch.LongTensor(self.token[idx]['attention_mask'])
        img_input = self.img_input[idx]
        img_mask = torch.tensor([1])

        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return img_input, img_mask, text_input,text_attention,label

class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        # self.classifier = nn.Linear(768, 2)
        self.fc1 = nn.Linear(3*768, 3*768)
        self.fc2 = nn.Linear(3*768, 768)
        self.classifier = nn.Linear(768, 2)

    def forward(self, frame_input, vedio_mask,title_input, text_mask):
        vision_embedding = self.visual_encoder(frame_input)
        bert_semantics_embedding, bert_feature = self.bert(input_ids= title_input, attention_mask=text_mask,return_dict=False) #768
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                   tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
        cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1) #768
        
        img_feature = vision_embedding.squeeze(1)
        feature_concat = torch.cat((bert_feature, img_feature,cross_attn_result_text),dim=1)  # new add
        feature_dnn = self.fc2(torch.relu(self.fc1(feature_concat)))
        prediction = self.classifier(feature_dnn)
  
        return prediction

 

if __name__ == '__main__':
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
    model = ConcatModel()
    epochs = 20
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')

    os.makedirs('model_dict/ConcatModel', exist_ok=True)
    save_model_path = 'model_dict/ConcatModel/best_f1.pickle'
    record_path = 'model_dict/ConcatModel/record.txt'

    # training
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SingleStream().to(device)
    model = torch.nn.parallel.DataParallel(model.to(device))
    
    learning_rate = 1e-6
    optimizer = Adam(model.parameters(), lr=learning_rate)
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

        for frame_input, vedio_mask,title_input, text_mask, label in tqdm(train_dataloader):
            sample_size_train+=1
            model.train()
            frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
            prediction = model(frame_input, vedio_mask,title_input, text_mask)
            loss, accuracy, _, _ = cal_loss(prediction,label)  
            epoch_loss_train += loss.item()
            epoch_acc_train += accuracy.item()
            loss.backward()
            optimizer.step()

        prediction_all = []
        label_all = []

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
        | f_1 Score: {f1_score_epoch: .3f}'''
        print(record)

        if f1_score_epoch > f1_score_best:
            torch.save(model.state_dict(), save_model_path)
            batch_size_record = str(batch_size)
            with open(record_path, "w") as file:
                file.write(record+batch_size_record)
            f1_score_best = f1_score_epoch
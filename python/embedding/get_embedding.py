import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose,Resize,CenterCrop,Normalize,ToTensor
import clip
from transformers import BertTokenizer
from tqdm import tqdm
import pickle


# image_embedding
class TransferToImage(Dataset):
    def __init__(self,data_path,modal_type):
        """
        data_path: "train_EEG.csv", "test_EEG.csv", "train_act.csv", "test_act.csv"
        modal_type: "EEG", "act"
        """
        self.tensor = []
        df = pd.read_csv(data_path)
        op_upsample = nn.Upsample(scale_factor=74,mode='nearest')
        pad = nn.ZeroPad2d(padding=(1,1,1,1))
        for i in range(len(df)):
            df_list = df.loc[i].tolist()
            if modal_type == "act":
                df_list = df_list + [df_list[-1]]*2
                df_tensor = torch.Tensor(df_list).reshape(3,3,3).permute(2,0,1).unsqueeze(0)
                df_tensor = pad(op_upsample(df_tensor)).squeeze(0) # torch.size(3,224,224)
                self.tensor.append(df_tensor)
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, index):
        return torch.FloatTensor(self.tensor[index])
    
def img_encode(data_path,modal_type,process_model,coef_model):
    """
    data_path: "train_EEG.csv", "test_EEG.csv", "train_act.csv", "test_act.csv"
    modal_type: "EEG", "act"
    process_model: "clip"
    coef_model: "ViT-B/32","ViT-B/16"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if process_model == "clip":
        model, preprocess = clip.load(coef_model)
        model = model.to(device)
        dataset = TransferToImage(data_path,modal_type)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False)
        feature=[]
        with torch.no_grad():
            for data in tqdm(dataloader):
                feature.append(model.encode_image(data.to(device)))
        return np.array(torch.cat(feature,dim=0).detach().cpu())
# !pwd python
# !mkdir embedding embedding/img embedding/txt

for data_path in ["../dataset/train_act.csv", "../dataset/test_act.csv"]:
    modal_type = "act"
    for process_model,coef_model in [['clip','ViT-B/32'],['clip','ViT-B/16']]:
        img_encode_array = img_encode(data_path,modal_type,process_model,coef_model)
        coef_model_path = coef_model.replace("/","_").replace("-","_")
        save_path = "img/ + modal_type" + "/" + process_model + "_" + coef_model_path + "_embedding.pickle"
        with open(save_path, 'wb') as f:
            img_encode_array.dump(f)



def text_feature(data_path,process_model,coef_model):
    """
    data_path: "train_EEG.csv", "test_EEG.csv", "train_action.csv", "test_action.csv"
    process_model: "bert"
    coef_model: "bert-base-uncased"
    """
    df = pd.read_csv(data_path)
    text_embedding=[]
    if process_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(coef_model)
        for i in range(len(df)):
            sentence = " ".join([str(i) for i in df.loc[1].tolist()])
            text_embedding.append(tokenizer(sentence,padding="max_length",truncation=True,max_length=512))
    return text_embedding

    


# action_img = img_feature("train_action.csv","action","clip","ViT-B/32")
# print(action_img.shape)
# with open("train_action_img.pickle", 'wb') as f:
#     action_img.dump(f)
    
    
    
class MultiModalDataset(Dataset):
    '''
    '''
    def __init__(self,eeg_data_path,act_data_path,label_path,
                 eeg_process_model, eeg_coef_model, act_process_model, act_coef_model, transfer_type):
        # self.eeg_df = pd.read_csv(eeg_df_path)
        # self.label = self.eeg_df['label']
        # with open(action_path, 'rb') as f:
        #     self.train_clip_feature = pickle.load(f)
        # with open(eeg_path, 'rb') as f:
        #     self.train_text_embedding = pickle.load(f)
        self.eeg_data_path = eeg_data_path
        self.act_data_path = act_data_path
        self.label = pd.read_csv(label_path)["label"]

        self.eeg_process_model = eeg_process_model
        self.eeg_coef_model = eeg_coef_model
        self.act_process_model = act_process_model
        self.act_coef_model = act_coef_model
        self.transfer_type = transfer_type

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        if self.transfer_type=="eeg2txt_act2img":
            eeg2tex_embedding = text_feature(self.eeg_data_path,self.eeg_process_model,self.eeg_coef_model)
            act2img_embedding = img_encode(self.act_data_path,"act",self.act_process_model,self.act_coef_model)
            eeg_input = torch.tensor(eeg2tex_embedding[idx]['input_ids']) # torch.Size([1, 512])
            eeg_mask = torch.tensor(eeg2tex_embedding[idx]['attention_mask'])
            action_input = torch.tensor(act2img_embedding[idx]).unsqueeze(0)# torch.Size([1, 1,512])
            action_mask = torch.tensor([1])

        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])

        return eeg_input,eeg_mask,action_input,action_mask,label

# train_dataset = MultiModalDataset(eeg_data_path = "train_EEG.csv",
#                                   act_data_path = "train_act.csv",
#                                   label_path = "train_label.csv",
#                                   eeg_process_model = "bert",
#                                   eeg_coef_model = "bert-base-uncased", 
#                                   act_process_model = "clip", 
#                                   act_coef_model = "ViT-B/32", 
#                                   transfer_type = "eeg2txt_act2img")
# print("1done")
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# print("2done")
# for eeg_input,eeg_mask,action_input,action_mask,label in tqdm(train_dataloader):
#     print(eeg_input.shape)
#     print(eeg_mask.shape)
#     print(action_input.shape)
#     print(action_mask.shape)
#     print(label.shape)



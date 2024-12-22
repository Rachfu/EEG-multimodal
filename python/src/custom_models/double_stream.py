import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import BertModel
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score
import os
from opacus import PrivacyEngine
import numpy as np

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

class DoubleStream_ti(nn.Module):
    def __init__(self,dp_mode,bert_coef):
        super().__init__()
        self.dp_mode = dp_mode
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(768*3, 2)
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,epsilon):
        device = self.device
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=act_img_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=act_img_mask==0, 
                                                    memory_key_padding_mask=eeg_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature,cross_attn_result),dim=1)
        if self.dp_mode == 'dropout_laplacian':
            pooled_output  = self.dropout(feature_concat)
            pooled_output_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output = (pooled_output - pooled_output_min) / (pooled_output_max - pooled_output_min)
            lap_sigma=1/epsilon
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
            noise = m.sample([pooled_output.shape[0]]).to(device)
            pooled_output += noise.view(-1, 1)
            feature_concat = pooled_output
        prediction = self.classifier(feature_concat)
        return prediction

class DoubleStream_tt(nn.Module):
    def __init__(self,dp_mode,bert_coef):
        super().__init__()
        self.dp_mode = dp_mode
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(768*3, 2)
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_txt_input,act_txt_mask,epsilon):
        device = self.device
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_txt_semantics_embedding, act_txt_feature = self.bert(input_ids= act_txt_input, 
                                                                 attention_mask=act_txt_mask,
                                                                 return_dict=False) #768
        
        cross_attn_result = self.multi_head_decoder(tgt=act_txt_semantics_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2))
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_txt_feature,cross_attn_result),dim=1)
        if self.dp_mode == 'dropout_laplacian':
            pooled_output  = self.dropout(feature_concat)
            pooled_output_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output = (pooled_output - pooled_output_min) / (pooled_output_max - pooled_output_min)
            lap_sigma=1/epsilon
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
            noise = m.sample([pooled_output.shape[0]]).to(device)
            pooled_output += noise.view(-1, 1)
            feature_concat = pooled_output
        prediction = self.classifier(feature_concat)
        return prediction

class DoubleStream_it(nn.Module):
    def __init__(self,dp_mode,bert_coef):
        super().__init__()
        self.dp_mode = dp_mode
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(768*3, 2)
        
    def forward(self, eeg_img_input,eeg_img_mask,act_txt_input,act_txt_mask,epsilon):
        device = self.device
        eeg_img_embedding = self.visual_encoder(eeg_img_input)
        eeg_img_feature = eeg_img_embedding.squeeze(1)
        act_txt_semantics_embedding, act_txt_feature = self.bert(input_ids= act_txt_input, 
                                                                 attention_mask=act_txt_mask,
                                                                 return_dict=False) #768
        cross_attn_result = self.multi_head_decoder(tgt=eeg_img_embedding.permute(1,0,2), 
                                                    memory=act_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=eeg_img_mask==0, 
                                                    memory_key_padding_mask=act_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_img_feature, act_txt_feature,cross_attn_result),dim=1)
        if self.dp_mode == 'dropout_laplacian':
            pooled_output  = self.dropout(feature_concat)
            pooled_output_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output = (pooled_output - pooled_output_min) / (pooled_output_max - pooled_output_min)
            lap_sigma=1/epsilon
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
            noise = m.sample([pooled_output.shape[0]]).to(device)
            pooled_output += noise.view(-1, 1)
            feature_concat = pooled_output
        prediction = self.classifier(feature_concat)
        return prediction

class DoubleStream_ii(nn.Module):
    def __init__(self,dp_mode):
        super().__init__()
        self.dp_mode = dp_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(768*3, 2)
        
    def forward(self, eeg_img_input,eeg_img_mask,act_img_input,act_img_mask,epsilon):
        device = self.device
        eeg_img_embedding = self.visual_encoder(eeg_img_input)
        eeg_img_feature = eeg_img_embedding.squeeze(1)
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=eeg_img_embedding.permute(1,0,2), 
                                                    memory=act_img_embedding.permute(1,0,2),)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_img_feature, act_img_feature,cross_attn_result),dim=1)
        if self.dp_mode == 'dropout_laplacian':
            pooled_output  = self.dropout(feature_concat)
            pooled_output_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
            pooled_output = (pooled_output - pooled_output_min) / (pooled_output_max - pooled_output_min)
            lap_sigma=1/epsilon
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
            noise = m.sample([pooled_output.shape[0]]).to(device)
            pooled_output += noise.view(-1, 1)
            feature_concat = pooled_output
        prediction = self.classifier(feature_concat)
        return prediction



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
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
import clip

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
     

    # # train_dataset = ConcatModelDataset('feature/train_EEG.csv','feature/new_train_img_tensor.pickle')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # token = tokenizer('Are you ok?',padding='max_length',truncation=True,max_length=512)
    # input_ids = torch.LongTensor(token['input_ids']).unsqueeze(0)
    # attention_mask = torch.LongTensor(token['attention_mask']).unsqueeze(0)

output_dim = 4  # 你可以根据自己的需求设置 output_dim 的值
self_w = torch.nn.parameter.Parameter(data=torch.rand(output_dim))
self_w_tensor = self_w.unsqueeze(1)  # 将 self.w 转换为形状为 (output_dim, 1) 的张量
one_minus_self_w_tensor = 1 - self_w_tensor  # 计算 1 - self.w 对应位置的值
result_tensor = torch.cat([self_w_tensor, one_minus_self_w_tensor], dim=1)  # 将两列张量按列拼接
# print(result_tensor)

def gumbel_dropout(input,w,tau=0.1,hard=True):
    w_tensor = w.unsqueeze(1)
    one_minus_w_tensor = 1 - w_tensor
    w_tensor = torch.cat([w_tensor, one_minus_w_tensor], dim=1)
    gumbel_w = F.gumbel_softmax(w_tensor,tau,hard)
    mask = gumbel_w[:, 1]
    return input*mask/(1.0-w)

def Lap_noise(input,epsilon):
    pooled_output_min = torch.min(input, dim=-1, keepdims=True)[0]
    pooled_output_max = torch.max(input, dim=-1, keepdims=True)[0]
    pooled_output = (input - pooled_output_min) / (pooled_output_max - pooled_output_min)
    lap_sigma=1/epsilon
    m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
    noise = m.sample([pooled_output.shape[0]])
    pooled_output += noise.view(-1, 1)

    # pooled_output += noise.squeeze(1)  # 后期考虑batch后可能改
    return pooled_output

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

class GumbelDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tau, epsilon):
        super(GumbelDropoutNet, self).__init__()
        self.w = torch.nn.parameter.Parameter(data=torch.rand(output_dim))
        self.dropout = GumbelSoftmaxDropout(tau)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 2) # without fc_layers
        self.epsilon = epsilon

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = self.fc2(x)
        res = self.dropout(x,self.w)
        res_lap = Lap_noise(res,self.epsilon)
        prediction = self.classifier(res_lap)
        return prediction




input_dim = 3*768
hidden_dim = 2*768
output_dim = 768
tau = 0.1
epsilon = 0.1
model = GumbelDropoutNet(input_dim, hidden_dim, output_dim, tau,epsilon)
# inputs = torch.rand(input_dim)
# labels = torch.tensor(0)
# pre = model(inputs)
# print(labels)
# print(pre)
# print(F.cross_entropy(pre, labels))
# def loss_function(output, target, model, weight_decay):
#     cross_entropy_loss = F.cross_entropy(output, target)
#     regularization_loss = weight_decay * torch.sum(model.w ** 2)
#     total_loss = cross_entropy_loss + regularization_loss
#     return total_loss

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# num_epochs = 2
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = loss_function(outputs, labels, model, weight_decay=0.5)
#     loss.backward()
#     optimizer.step()
#     total_loss += loss.item()
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()



    
    

# print(labels)




# feature_test = torch.nn.parameter.Parameter(data=torch.rand(4)) # feature = 4
# # dropout = nn.Dropout(0.4)
# self_w = torch.nn.parameter.Parameter(data=torch.rand(4))
# print(self_w)
# print(gumbel_dropout(feature_test,self_w))

# self_w_tensor = self_w.unsqueeze(1)  # 将 self.w 转换为形状为 (output_dim, 1) 的张量
# one_minus_self_w_tensor = 1 - self_w_tensor  # 计算 1 - self.w 对应位置的值
# w_tensor = torch.cat([self_w_tensor, one_minus_self_w_tensor], dim=1)  # 将两列张量按列拼接
# print(w_tensor)
# gumbel_w = F.gumbel_softmax(w_tensor,tau=0.1,hard= True) # tau越小越接近argmax
# print(gumbel_w)
# mask = gumbel_w[:, 1]
# print(mask)
# print(feature_test)
# print(feature_test*mask/self_w)
# result_tensor = dropout_test(feature_test,0.4)
# print(result_tensor)

#################################################################
batch_size = 8
train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

bert = BertModel.from_pretrained('bert-base-uncased')
visual_encoder = nn.Linear(512, 768)
bert_output_size = 768
multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
multi_head_decoder = TransformerDecoder(multi_head_decoderlayer, num_layers=3)
# self.fc1 = nn.Linear(3*768, 3*768)
# self.fc2 = nn.Linear(3*768, 768)
fc_layers = nn.Sequential(
    nn.Linear(3*768, 3*768),
    nn.ReLU(),
    nn.Linear(3*768, 768),
    nn.Tanh(),
)
# classifier = nn.Linear(768, 2)
# dropout = nn.Dropout(0.4)
weight_decay = 0.5
def loss_function(outputs, label, model, weight_decay):
    cross_entropy_loss = F.cross_entropy(outputs, label.squeeze())
    tmp = (1-model.w)*np.exp(epsilon)+model.w
    loss_w,_ = torch.max(tmp, dim=0)
    total_loss = (1-weight_decay)*cross_entropy_loss + weight_decay *loss_w
    return total_loss

for frame_input, vedio_mask,title_input, text_mask, label in train_dataloader:
    vision_embedding = visual_encoder(frame_input)
    bert_semantics_embedding, bert_feature = bert(input_ids= title_input, attention_mask=text_mask,return_dict=False) #768
    cross_attn_result_text = multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
    cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1) #768
    
    img_feature = vision_embedding.squeeze(1)
    feature_concat = torch.cat((bert_feature, img_feature,cross_attn_result_text),dim=1)  # new add
    # print(feature_concat.shape)
    # outputs = model(feature_concat)
    # print(outputs.shape)
    # print(label.shape)
    # loss = F.cross_entropy(outputs, label.squeeze())
    # tmp = (1-model.w)*np.exp(epsilon)+model.w
    # loss_w,_ = torch.max(tmp, dim=0)
    # # print(torch.max(tmp, dim=0))
    # print(loss)
    # print(loss_w)
    # loss = loss_function(outputs, label, model, weight_decay)
    # print(loss)


#     pooled_output  = dropout(feature_concat)
#     print(pooled_output.shape)
#     pooled_output_min = torch.min(pooled_output, dim=-1, keepdims=True)[0]
#     pooled_output_max = torch.max(pooled_output, dim=-1, keepdims=True)[0]
#     pooled_output = (pooled_output - pooled_output_min) / (pooled_output_max - pooled_output_min)
#     print(pooled_output.shape)
#     EPSILON = 7.5 ## 暂时设一个
#     lap_sigma=1/EPSILON
#     m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
#     noise = m.sample([pooled_output.shape[0]])
#     print(noise.shape)
#     pooled_output += noise.view(-1, 1)
#     print(pooled_output.shape)
#     # feature_dnn = self.fc2(torch.relu(self.fc1(feature_concat)))
    break

optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    outputs = model(feature_concat)
    loss = loss_function(outputs, label, model, weight_decay)
    loss.backward(retain_graph=True)
    optimizer.step()


    ############### 问题：每次initial一个新的w，会有问题吗？应该不会有，因为就跑了一次init
    # total_loss += loss.item()
    # _, predicted = torch.max(outputs.data, 1)
    # total += labels.size(0)
    # correct += (predicted == labels).sum().item()






    

    
        

    
    
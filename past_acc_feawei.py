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
import matplotlib.pyplot as plt
import scipy.stats as stats


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda', 1)
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

class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        # self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.DP = nn.Parameter(torch.cat((torch.full((1, 768), 0.4), torch.full((1, 768), 0.3), torch.full((1, 768), 0.5)), dim=1))

        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        self.eps = torch.tensor(1.0) # pre:1.0,0.1

    def forward(self, frame_input, vedio_mask, title_input, text_mask,hard):
        vision_embedding = self.visual_encoder(frame_input)
        bert_semantics_embedding, bert_feature = self.bert(input_ids= title_input, 
                                                           attention_mask=text_mask,
                                                           return_dict=False) #768
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), 
                                                         memory=bert_semantics_embedding.permute(1,0,2),
                                                         tgt_key_padding_mask=vedio_mask == 0, 
                                                         memory_key_padding_mask=text_mask == 0)
        cross_attn_result_text = cross_attn_result_text.permute(1, 0, 2).mean(dim=1) #768
        
        img_feature = vision_embedding.squeeze(1)
        feature_concat = torch.cat((bert_feature, img_feature, cross_attn_result_text), dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        # feature = feature.detach().cpu().numpy()
        # print(test.shape)
        # reshaped_data = test.reshape(8,3, 768)
        # mean_values = np.mean(reshaped_data, axis=2)
        # print(mean_values)
        return feature


def main2():
    '''
    Adding noise with Lap to all features
    '''
    weight =np.empty((0, 2304))
    batch_size = 8
    train_dataset = MultiModalDataset_ti('feature/train_EEG.csv','feature/action/train_clip_v2.pickle','feature/EEG/train_bert.pickle')
    val_dataset = MultiModalDataset_ti('feature/test_EEG.csv','feature/action/test_clip_v2.pickle','feature/EEG/test_bert.pickle')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ConcatModel()
    model = model.to(device)
    for frame_input, vedio_mask,title_input, text_mask, label in tqdm(train_dataloader):
        frame_input, vedio_mask,title_input, text_mask, label = frame_input.to(device), vedio_mask.to(device),title_input.to(device), text_mask.to(device), label.to(device)
        feature = model(frame_input, vedio_mask,title_input, text_mask,hard = False)
        feature = feature.detach().cpu().numpy()
        weight = np.vstack((weight, feature))
        # print(weight.shape)

    with open('feawei.pkl', 'wb') as f:
        pickle.dump(weight, f)


if __name__ == '__main__':
    # main2()
    with open('feawei.pkl', 'rb') as f:
        weight = pickle.load(f)
        # print(weight.shape) # (2402, 2304)
        mean_values = np.mean(weight, axis=0)
        # mean_values = (mean_values-np.mean(mean_values))/np.std(mean_values)
        # DP_init = nn.Parameter(torch.cat((torch.full((1, 768), 0.4), torch.full((1, 768), 0.5), torch.full((1, 768), 0.3)), dim=1)) # not bad init; results in newfrac_1.0eps_tt

        # # print(mean_values.shape) #(2304,)
        # k=5
        # w_init = 1-F.sigmoid(torch.tensor(k * (mean_values), dtype=torch.float32))
        # DP_tt = DP_init+nn.Parameter(w_init.unsqueeze(0))-0.5
        # print(DP_tt)
        # # test = w_init.detach().cpu().numpy()
        # # # print(test.shape)
        # # reshaped_data = test.reshape(3, 768)
        # # mean_values = np.mean(reshaped_data, axis=1)
        # # print(mean_values)
        mo_w = mean_values[0:768]
        plt.figure(figsize=(7.5, 5))  # 设置图形的大小
        plt.hist(mo_w, bins=30, alpha=0.75, density=True,edgecolor='black')  # bins参数控制直方图的条形数
        kde = stats.gaussian_kde(mo_w)
        x = np.linspace(min(mo_w), max(mo_w), 1000)
        kde_values = kde(x)

        # 绘制KDE曲线
        plt.plot(x, kde_values, color='red', linestyle='-', linewidth=2)
        # plt.plot(bin_centers, counts, linestyle='-', marker='o', color='blue')  # 线性和标记点
        plt.title('Distribution of the element-wise dropout rate of EEG feature')  # 设置标题
        plt.xlabel('Dropout rate')  # 设置X轴标签
        plt.ylabel('Frequency')  # 设置Y轴标签
        mean_value = np.mean(mo_w)
        plt.axvline(x=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')


        # plt.grid(True)  # 显示网格
        plt.savefig('tt.pdf') 
        plt.close()


        # print(w_init)
    

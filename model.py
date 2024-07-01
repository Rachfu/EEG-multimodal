import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import BertModel


def get_model(cfg):
    if cfg.data_name == 'EEG':
        model = ConcatModel()
    model.eps = torch.tensor(cfg.eps).cuda()
    return model.cuda()

class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(768 * 3, 2) # without fc_layers
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))

    def feature(self, x):
        frame_input, vedio_mask, title_input, text_mask = x
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
        feature_output = (feature_concat - feature_min) / (feature_max - feature_min)
        return feature_output

    def forward(self, x, hard=True):
        w = F.sigmoid(self.DP)
        feature = self.feature(x)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).cuda()
        eps_hat = ((self.eps.exp() - w) / (1 - w)).log()
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        prediction = self.classifier(feature)
        return prediction
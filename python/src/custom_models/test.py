from dataset import MultiModalDataset_ti
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from double_stream import DoubleStream_ti


def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

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

# batch_size = 8
# train_dataset = MultiModalDataset_ti('data/embedding/EEG/txt/bert_bert_base_uncased/train.pickle',
#                                      'data/embedding/act/img/clip_ViT_B_32/train.pickle',
#                                      'data/processed/train_label.csv')
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# with torch.no_grad():
#     for eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,label in tqdm(train_dataloader):
#         print(eeg_txt_input.shape)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from dataset import MultiModalDataset_ti,MultiModalDataset_tt,MultiModalDataset_it,MultiModalDataset_ii
from double_stream import DoubleStream_ti,DoubleStream_tt,DoubleStream_it,DoubleStream_ii
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import warnings
from sklearn.metrics import f1_score
from opacus import PrivacyEngine
import time

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

warnings.filterwarnings('ignore')

class TrainAndTest(object):
    def __init__(self,
                 batch_size=8,
                 learning_rate= 1e-6,
                 epochs = 50,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cal_loss(self,prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    def train(self,multimodal_type,dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,cross_atn_type,epsilon):
        """
        multimodal_type = "ti","tt","it","ii"
        dp_mode: "feature_all_lap"
        """
        batch_size = self.batch_size
        eeg_model_coef_standardized = eeg_model_coef.replace("/","_").replace("-","_")
        act_model_coef_standardized = act_model_coef.replace("/","_").replace("-","_")
        if cross_atn_type == "double_stream":
            if multimodal_type == "ti":
                eeg_txt_path = "data/embedding/EEG/txt/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
                act_img_path = "data/embedding/act/img/" + act_model + "_" + act_model_coef_standardized + "/"
                label_path = "data/processed/"
                train_dataset = MultiModalDataset_ti(eeg_txt_path + "train.pickle",
                                                    act_img_path + "train.pickle",
                                                    label_path+"train_label.csv")
                test_dataset = MultiModalDataset_ti(eeg_txt_path + "test.pickle",
                                                    act_img_path + "test.pickle",
                                                    label_path+"test_label.csv")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                model = DoubleStream_ti(dp_mode = dp_mode,bert_coef = eeg_model_coef)
            if multimodal_type == "tt":
                eeg_txt_path = "data/embedding/EEG/txt/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
                act_txt_path = "data/embedding/act/txt/" + act_model + "_" + act_model_coef_standardized + "/"
                label_path = "data/processed/"
                train_dataset = MultiModalDataset_tt(eeg_txt_path + "train.pickle",
                                                    act_txt_path + "train.pickle",
                                                    label_path+"train_label.csv")
                test_dataset = MultiModalDataset_tt(eeg_txt_path + "test.pickle",
                                                    act_txt_path + "test.pickle",
                                                    label_path+"test_label.csv")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                model = DoubleStream_tt(dp_mode = dp_mode,bert_coef = eeg_model_coef)
            if multimodal_type == "it":
                eeg_img_path = "data/embedding/EEG/img/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
                act_txt_path = "data/embedding/act/txt/" + act_model + "_" + act_model_coef_standardized + "/"
                label_path = "data/processed/"
                train_dataset = MultiModalDataset_it(eeg_img_path + "train.pickle",
                                                    act_txt_path + "train.pickle",
                                                    label_path+"train_label.csv")
                test_dataset = MultiModalDataset_it(eeg_img_path + "test.pickle",
                                                    act_txt_path + "test.pickle",
                                                    label_path+"test_label.csv")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                model = DoubleStream_it(dp_mode = dp_mode,bert_coef = act_model_coef)
            if multimodal_type == "ii":
                eeg_img_path = "data/embedding/EEG/img/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
                act_img_path = "data/embedding/act/img/" + act_model + "_" + act_model_coef_standardized + "/"
                label_path = "data/processed/"
                train_dataset = MultiModalDataset_ii(eeg_img_path + "train.pickle",
                                                    act_img_path + "train.pickle",
                                                    label_path+"train_label.csv")
                test_dataset = MultiModalDataset_ii(eeg_img_path + "test.pickle",
                                                    act_img_path + "test.pickle",
                                                    label_path+"test_label.csv")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                model = DoubleStream_ii(dp_mode = dp_mode)

        learning_rate = self.learning_rate
        epochs = self.epochs
        optimizer = Adam(model.parameters(), lr=learning_rate)
        device = self.device
        model = model.to(device)

        model_path = "models/custom/" + cross_atn_type +"/"+multimodal_type+"/"+eeg_model_coef_standardized+"&"+act_model_coef_standardized+"/" + str(epsilon)+"/"
        log_path = "logs/" + cross_atn_type +"/"+multimodal_type+"/"+eeg_model_coef_standardized+"&"+act_model_coef_standardized+"/" + str(epsilon)+"/"
        for path in [model_path,log_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        save_model_path = model_path + "best_f1.pickle"
        whole_log_path = log_path+"whole_record.txt"
        best_log_path = log_path+"best_record.txt"
        
        f1_score_best = 0.5

        # training
        for epoch in range(epochs):
            start_time = time.time()
            epoch_acc_train = 0
            epoch_loss_train = 0
            epoch_acc_test = 0
            epoch_loss_test = 0
            sample_size_train = 0
            sample_size_test = 0

            model.train()
            for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
                sample_size_train+=1
                model.train()
                optimizer.zero_grad()
                eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                if multimodal_type == "ti":
                    prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                elif multimodal_type == "it":
                    prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input,act_mask,epsilon)
                elif multimodal_type == "ii":
                    prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                else:
                    prediction = model(eeg_input,eeg_mask,act_input,act_mask,epsilon)
                loss, accuracy, _, _ = self.cal_loss(prediction,label)  
                epoch_loss_train += loss.item()
                epoch_acc_train += accuracy.item()
                loss.backward()
                optimizer.step()

            prediction_all = []
            label_all = []
            model.eval()
            with torch.no_grad():
                for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
                    sample_size_test +=1
                    eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                    if multimodal_type == "ti":
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                    elif multimodal_type == "it":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input,act_mask,epsilon)
                    elif multimodal_type == "ii":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                    else:
                        prediction = model(eeg_input,eeg_mask,act_input,act_mask,epsilon)
                    loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
                    prediction_all.extend(pred_label_id.cpu().numpy())
                    label_all.extend(label_id.cpu().numpy())
                    epoch_loss_test += loss.item()
                    epoch_acc_test += accuracy.item()

            f1_score_epoch = f1_score(prediction_all,label_all)
            end_time = time.time()
            time_cost = end_time-start_time

            record = f'''Epochs: {epoch + 1}
            | Train Loss: {epoch_loss_train/sample_size_train: .3f}
            | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
            | Test Loss: {epoch_loss_test/sample_size_test: .3f}
            | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
            | f_1 Score: {f1_score_epoch: .3f}
            | Time Cost: {time_cost: .1f}\n'''
            print(record)
            with open(whole_log_path, "a") as file:
                file.write(record)

            if f1_score_epoch > f1_score_best:
                torch.save(model.state_dict(), save_model_path)
                f1_best_record = record
                f1_score_best = f1_score_epoch
                with open(best_log_path, "w") as file:
                    file.write(f1_best_record)





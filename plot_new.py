import matplotlib.pyplot as plt
import torch
from past_acc import ConcatModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
import pickle

####################################### figure_with_epoch #######################################
def main_epoch():
    def data_extraction(path,type):
        train = []
        val = []
        with open(path, 'r') as file:
            for line in file:
                if type == 'acc':
                    if "Train Accuracy" in line:
                        train.append(float(line.split(':')[-1].strip()))
                    elif "Val Accuracy" in line:
                        val.append(float(line.split(':')[-1].strip()))
                if type == 'loss':
                    if "Train Loss" in line:
                        train.append(float(line.split(':')[-1].strip()))
                    elif "Val Loss" in line:
                        val.append(float(line.split(':')[-1].strip()))
        return (train,val)

    path_1 = 'model_dict/new_10.0eps/whole_record.txt'
    path_2 = 'model_dict/newfrac_0.1eps/whole_record.txt'
    path_3 = 'model_dict/newfrac_1.0eps/whole_record.txt'

    data = [
        (data_extraction(path_1,'acc'), data_extraction(path_1,'loss')),
        (data_extraction(path_2,'acc'), data_extraction(path_2,'loss')),
        (data_extraction(path_3,'acc'), data_extraction(path_3,'loss'))
    ]
    e_list = [0.01,0.1,1.0]
    epoch = list(range(50))
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    for i, (acc, loss) in enumerate(data):
        train_acc, val_acc = acc
        train_loss, val_loss = loss

        axs[i,0].plot(epoch, train_acc, label='Train Accuracy',color = '#FF928B',linewidth=2) #B3D9FB
        axs[i,0].plot(epoch, val_acc, label='Val Accuracy', color = '#B3D9FB',linewidth=2)
        max_train_idx = train_acc.index(max(train_acc))
        max_val_idx = val_acc.index(max(val_acc))
        axs[i,0].scatter(max_train_idx, train_acc[max_train_idx], color='#FE6F5E', s=100, marker='*', zorder=5)
        axs[i,0].scatter(max_val_idx, val_acc[max_val_idx], color='#72A0C1', s=100, marker='*', zorder=5)
        axs[i,0].set_title(f'Accuracy with $\\epsilon$ = {e_list[i]}')
        axs[i,0].set_xlabel('Epoch')
        axs[i,0].set_ylabel('Accuracy') 
        axs[i,0].legend(loc='lower right')
        # axs[i,0].set_ylim(0.6,1.0)
        axs[i,0].grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0', alpha=0.5)  # 自定义网格线样式

        axs[i,1].plot(epoch, train_loss, label='Train Loss',color = '#FF928B',linewidth=2) #B3D9FB
        axs[i,1].plot(epoch, val_loss, label='Val Loss', color = '#B3D9FB',linewidth=2)
        axs[i,1].set_title(f'Loss with $\\epsilon$ = {e_list[i]}')
        axs[i,1].set_xlabel('Epoch')
        axs[i,1].set_ylabel('Loss') 
        axs[i,1].legend(loc='upper right')
        # axs[i,1].set_ylim(0.6,1.0)
        axs[i,1].grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0', alpha=0.5)  # 自定义网格线样式
    plt.tight_layout()
    plt.savefig('plot_new/cp4_fig1.pdf') 
    plt.close()

####################################### figure_with_feature #######################################
def feature():
    model = ConcatModel()
    model.load_state_dict(torch.load('model_dict/newfrac_1.0eps_newinit_k1/best_f1.pickle'))
    model.eval()
    w = F.sigmoid(model.DP.data).cpu().numpy().reshape(3, 768)  # 假设DP是模型的一个参数
    with open('feawei.pkl', 'rb') as f:
        weight = pickle.load(f)
    mean_values = np.mean(weight, axis=0)
    # fea_mag = mean_values.reshape(3, 768)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    def plot_fun(pos_x,pos_y,w,color,title,xlabel,kde_color,mean=True,color_mean = 'red'):
        axs[pos_x,pos_y].hist(w, bins=30, alpha=0.75, density=True,edgecolor='black',color= color)
        kde = stats.gaussian_kde(w)
        x = np.linspace(min(w), max(w), 1000)
        kde_values = kde(x)
        axs[pos_x,pos_y].plot(x, kde_values, color=kde_color, linestyle='-', linewidth=2)
        axs[pos_x,pos_y].set_title(title)  # 设置标题
        axs[pos_x,pos_y].set_xlabel(xlabel)  # 设置X轴标签
        axs[pos_x,pos_y].set_ylabel('Frequency')  # 设置Y轴标签
        if mean==True:
            mean_value = np.mean(w)
            axs[pos_x,pos_y].axvline(x=mean_value, color=color_mean, linestyle='--', label=f'Mean: {mean_value:.2f}')

    plot_fun(0,0,w[0,:],'#5F9C61',f'Dropout rate with EEG (Avg. = {np.mean(w[0,:]):.3f})','Dropout rate','#2C6344',mean=True,color_mean = 'black')
    plot_fun(1,0,w[1,:],'#B092B6',f'Dropout rate with OM (Avg. = {np.mean(w[1,:]):.3f})','Dropout rate','#61496D',mean=True,color_mean = 'black')
    plot_fun(2,0,w[2,:],'#E38D26',f'Dropout rate with CM (Avg. = {np.mean(w[2,:]):.3f})','Dropout rate','#C74D26',mean=True,color_mean = 'black')

    plot_fun(0,1,mean_values[0:768],'#A4C97C','Feature magnitude of EEG','Feature magnitude','#2C6344',mean=False)
    plot_fun(1,1,mean_values[768:768*2],'#CAC1D4','Feature magnitude of OM','Feature magnitude','#61496D',mean=False)
    plot_fun(2,1,mean_values[768*2:768*3],'#F1CC74','Feature magnitude of CM','Feature magnitude','#C74D26',mean=False)

    plt.tight_layout()
    plt.savefig('plot_new/cp4_fig2.pdf') 
    plt.close()

def acc_best():
    eps = ['0.01', '0.05', '0.1', '0.5', '1.0', '5.0']
    acc = [0.806, 0.892, 0.956, 0.968, 0.987, 0.988]

    # 使用 Matplotlib 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图形的尺寸
    plt.plot(eps, acc, marker='o', linestyle='-', color='#B3D9FB')  # 折线图，带有标记

    # 添加标题和标签
    plt.title('Best validation accuracy vs privacy budget')
    plt.xlabel('Privacy budget')
    plt.ylabel('Accuracy')

    # 可以添加网格以便更好地阅读
    plt.grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0', alpha=0.5)  # 自定义网格线样式
    plt.savefig('plot_new/cp4_fig3.pdf') 
    plt.close()



if __name__ == '__main__':
    # main_epoch()
    # feature()
    acc_best()
    
        

    




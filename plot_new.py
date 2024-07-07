import matplotlib.pyplot as plt
import torch
from past_acc import ConcatModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
import pickle
import os
from matplotlib.colors import LogNorm

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

def eps_epoch():
    epsilon_list = np.logspace(np.log10(0.01), np.log10(5.0), 20)
    epsilon_list = np.around(epsilon_list, decimals=3)

    epochs = 50
    val_accuracies = []

    # 读取每个子文件夹中的数据
    for epsilon in epsilon_list:
        folder_name = f"model_dict/eps_experiment/{epsilon}"
        file_path = os.path.join(folder_name, "whole_record.txt")
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue
        
        val_acc = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Train Accuracy" in line:  # revised for train and val
                    val_acc_value = float(line.split(":")[1].strip())
                    val_acc.append(val_acc_value)
        
        if len(val_acc) == epochs:
            val_accuracies.append(val_acc)
        else:
            print(f"File {file_path} does not contain {epochs} epochs of data.")
    plt.figure(figsize=(10, 6))

# 使用渐变颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_list)))

    for i, (val_acc, epsilon) in enumerate(zip(val_accuracies, epsilon_list)):
        # plt.plot(range(1, epochs + 1), val_acc, label=f"eps=exp({np.exp(epsilon):.2f})", color=colors[i])
        plt.plot(range(1, epochs + 1), val_acc, label=f"eps={epsilon}", color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Train accuracy')
    plt.title('Train accuracy over epochs for different privacy budget')
    plt.legend(loc='best', fontsize='small', ncol=2)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Privacy budget')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=LogNorm(vmin=epsilon_list.min(), vmax=epsilon_list.max())), label='Privacy budget (log scale)')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Privacy budget')
    plt.grid(True)
    plt.savefig('plot_new/cp4_fig4.pdf') 
    # plt.savefig('tt.pdf') 
    plt.close()
            
def eps_best():
    epsilon_list = np.logspace(np.log10(0.01), np.log10(5.0), 20)
    epsilon_list = np.around(epsilon_list, decimals=3)

    # 准备存储每个实验的最佳 Val Accuracy 和第10个epoch的 Val Accuracy
    best_val_accuracies = []
    epoch10_val_accuracies = []

    # 读取每个子文件夹中的数据
    for epsilon in epsilon_list:
        folder_name = f"model_dict/eps_experiment/{epsilon}"
        
        # 读取 best_record.txt 中的最佳 Val Accuracy
        best_file_path = os.path.join(folder_name, "best_record.txt")
        if not os.path.exists(best_file_path):
            print(f"File {best_file_path} does not exist.")
            best_val_accuracies.append(None)
        else:
            with open(best_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if "Val Accuracy" in line:
                        val_acc_value = float(line.split(":")[1].strip())
                        best_val_accuracies.append(val_acc_value)
                        break

        # 读取 whole_record.txt 中第10个epoch的 Val Accuracy
        whole_file_path = os.path.join(folder_name, "whole_record.txt")
        if not os.path.exists(whole_file_path):
            print(f"File {whole_file_path} does not exist.")
            epoch10_val_accuracies.append(None)
        else:
            with open(whole_file_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if "Epochs: 10" in lines[i]:
                        for j in range(i, len(lines)):
                            if "Val Accuracy" in lines[j]:
                                val_acc_value = float(lines[j].split(":")[1].strip())
                                epoch10_val_accuracies.append(val_acc_value)
                                break
                        break

    # 检查是否成功读取了所有数据
    if len(best_val_accuracies) != len(epsilon_list):
        print("Warning: Some best Val Accuracy data might be missing.")
    if len(epoch10_val_accuracies) != len(epsilon_list):
        print("Warning: Some epoch10 Val Accuracy data might be missing.")

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制最佳 Val Accuracy 曲线
    plt.plot(epsilon_list, best_val_accuracies, marker='o', linestyle='-', color='#87CEEB', label='Best Val Accuracy within 50 Epoches')

    # 绘制第10个epoch的 Val Accuracy 曲线
    plt.plot(epsilon_list, epoch10_val_accuracies, marker='x', linestyle='--', color='#2774AE', label='Val Accuracy at Epoch 10')

    plt.xscale('log')
    plt.xlabel('Privacy budget')
    plt.ylabel('Validation accuracy')
    plt.title('Validation accuracy for different privacy budget')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('plot_new/cp4_fig6.pdf') 
    plt.close()



if __name__ == '__main__':
    # main_epoch()
    # feature()
    # acc_best()
    # eps_epoch()
    eps_best()
    
        

    




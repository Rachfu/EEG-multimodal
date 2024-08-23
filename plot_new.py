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

        axs[i,0].plot(epoch, train_acc, label='Train Accuracy',color = '#FF928B',linewidth=4) #B3D9FB
        axs[i,0].plot(epoch, val_acc, label='Val Accuracy', color = '#B3D9FB',linewidth=4)  # linewidth changes from 2 to 4
        max_train_idx = train_acc.index(max(train_acc))
        max_val_idx = val_acc.index(max(val_acc))
        axs[i,0].scatter(max_train_idx, train_acc[max_train_idx], color='#FE6F5E', s=100, marker='*', zorder=10) # zorder changes from 5 to 8
        axs[i,0].scatter(max_val_idx, val_acc[max_val_idx], color='#72A0C1', s=100, marker='*', zorder=10)
        axs[i,0].set_title(f'Accuracy with $\\epsilon$ = {e_list[i]}', fontsize = 14)
        axs[i,0].set_xlabel('Epoch',fontsize = 12)
        axs[i,0].set_ylabel('Accuracy',fontsize = 12) 
        axs[i,0].legend(loc='lower right',fontsize = 12)
        # axs[i,0].set_ylim(0.6,1.0)
        axs[i,0].grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0', alpha=0.5)  # 自定义网格线样式

        axs[i,1].plot(epoch, train_loss, label='Train Loss',color = '#FF928B',linewidth=4) #B3D9FB
        axs[i,1].plot(epoch, val_loss, label='Val Loss', color = '#B3D9FB',linewidth=4)
        axs[i,1].set_title(f'Loss with $\\epsilon$ = {e_list[i]}',fontsize = 14)
        axs[i,1].set_xlabel('Epoch',fontsize = 12)
        axs[i,1].set_ylabel('Loss',fontsize = 12) 
        axs[i,1].legend(loc='upper right',fontsize = 12)
        # axs[i,1].set_ylim(0.6,1.0)
        axs[i,1].grid(True, linestyle='--', linewidth=0.5, color='#C0C0C0', alpha=0.5)  # 自定义网格线样式
    plt.tight_layout()
    plt.savefig('plot_new/cp4_fig1_new.pdf') 
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
                # if "Train Accuracy" in line:  # revised for train and val
                if "Val Accuracy" in line:  # revised for train and val
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

    plt.xlabel('Epoch',fontsize=12)
    # plt.ylabel('Train accuracy',fontsize=12)
    # plt.title('Train accuracy over epochs for different privacy budget',fontsize=14)
    plt.ylabel('Validation accuracy',fontsize=12)
    plt.title('Validation accuracy over epochs for different privacy budget',fontsize=14)

    plt.legend(loc='best', fontsize='small', ncol=2)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Privacy budget')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=LogNorm(vmin=epsilon_list.min(), vmax=epsilon_list.max())), label='Privacy budget (log scale)')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=LogNorm(vmin=epsilon_list.min(), vmax=epsilon_list.max())))
    cbar.set_label('Privacy budget (log scale)', fontsize=12)
    
    
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Privacy budget')
    plt.grid(True)
    # plt.savefig('plot_new/cp4_fig4_new.pdf') 
    plt.savefig('plot_new/cp4_fig5_new.pdf') 
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
    plt.plot(epsilon_list, best_val_accuracies, marker='o', linestyle='-', color='#87CEEB', label='Best Val Accuracy within 50 Epoches',linewidth=2,markersize=8)

    # 绘制第10个epoch的 Val Accuracy 曲线
    plt.plot(epsilon_list, epoch10_val_accuracies, marker='x', linestyle='--', color='#2774AE', label='Val Accuracy at Epoch 10',linewidth=2,markersize=8)

    plt.xscale('log')
    plt.xlabel('Privacy budget',fontsize=12)
    plt.ylabel('Validation accuracy',fontsize=12)
    plt.title('Validation accuracy for different privacy budget',fontsize=14)
    plt.legend(loc='best',fontsize=12)
    plt.grid(True)

    plt.savefig('plot_new/cp4_fig6_new.pdf') 
    plt.close()


def feature_new():
    model = ConcatModel(epsilon=1.0)
    model.load_state_dict(torch.load('model_dict/newfrac_1.0eps_newinit_k1/best_f1.pickle'))
    model.eval()
    

    w = F.sigmoid(model.DP.data).cpu().numpy().squeeze(0)
    # w = F.sigmoid(model.DP.data).cpu().numpy().reshape(3, 768)  # 假设DP是模型的一个参数
    with open('feawei.pkl', 'rb') as f:
        weight = pickle.load(f)
    mean_values = np.mean(weight, axis=0)

    w_split = np.split(w, 3)
    mean_values_split = np.split(mean_values, 3)

    sorted_w = []
    sorted_mean_values = []

    for w_part, mean_part in zip(w_split, mean_values_split):
        combined = list(zip(w_part, mean_part))
        sorted_combined = sorted(combined)
        sorted_w_part, sorted_mean_part = zip(*sorted_combined)
        sorted_w.extend(sorted_w_part)
        sorted_mean_values.extend(sorted_mean_part)

    w = np.array(sorted_w)
    mean_values = np.array(sorted_mean_values)
    variance = 1/(np.log((np.exp(1.0)-w)/(1-w)))
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    def plot_fun(pos_x, pos_y, vector, color, title, ylabel, mean_color='black', show_mean=True):
        axs[pos_x, pos_y].scatter(range(len(vector)),vector, color=color,s=5)
        if show_mean:
            axs[pos_x, pos_y].axhline(y=np.mean(vector), color=mean_color, linestyle='--')
        axs[pos_x, pos_y].set_xlabel('Index',fontsize=12)
        axs[pos_x, pos_y].set_ylabel(ylabel,fontsize=12)
        axs[pos_x, pos_y].set_title(title)
        axs[pos_x, pos_y].set_ylim(0, 1)
        axs[pos_x, pos_y].grid(True)
        # axs[pos_x, pos_y].tick_params(axis='both', which='major', labelsize=10)

    # def plot_fun(pos_x,pos_y,w,color,title,xlabel,kde_color,mean=True,color_mean = 'red'):
    #     axs[pos_x,pos_y].hist(w, bins=30, alpha=0.75, density=True,edgecolor='black',color= color)
    #     kde = stats.gaussian_kde(w)
    #     x = np.linspace(min(w), max(w), 1000)
    #     kde_values = kde(x)
    #     axs[pos_x,pos_y].plot(x, kde_values, color=kde_color, linestyle='-', linewidth=2)
    #     axs[pos_x,pos_y].set_title(title)  # 设置标题
    #     axs[pos_x,pos_y].set_xlabel(xlabel)  # 设置X轴标签
    #     axs[pos_x,pos_y].set_ylabel('Frequency')  # 设置Y轴标签
    #     if mean==True:
    #         mean_value = np.mean(w)
    #         axs[pos_x,pos_y].axvline(x=mean_value, color=color_mean, linestyle='--', label=f'Mean: {mean_value:.2f}')
    
    plot_fun(0,0,w[0:768],'#5F9C61',f'Dropout rate with sorted EEG features (Avg. = {np.mean(w[0:768]):.3f})','Dropout rate',mean_color = '#2C6344',show_mean=True)
    plot_fun(1,0,w[768:768*2],'#B092B6',f'Dropout rate with sorted OM features (Avg. = {np.mean(w[768:768*2]):.3f})','Dropout rate','#61496D',True)
    plot_fun(2,0,w[768*2:768*3],'#E38D26',f'Dropout rate with sorted CM features (Avg. = {np.mean(w[768*2:768*3]):.3f})','Dropout rate','#C74D26',True)

    plot_fun(0,1,variance[0:768],'#5F9C61',f'Laplacian noise scale with sorted EEG features (Avg. = {np.mean(variance[0:768]):.3f})','Laplacian noise scale',mean_color = '#2C6344',show_mean=True)
    plot_fun(1,1,variance[768:768*2],'#B092B6',f'Laplacian noise scale with sorted OM features (Avg. = {np.mean(variance[768:768*2]):.3f})','Laplacian noise scale','#61496D',True)
    plot_fun(2,1,variance[768*2:768*3],'#E38D26',f'Laplacian noise scale with sorted CM features (Avg. = {np.mean(variance[768*2:768*3]):.3f})','Laplacian noise scale','#C74D26',True)


    plot_fun(0,2,mean_values[0:768],'#A4C97C',f'Magnitude of sorted EEG features (Avg. = {np.mean(mean_values[0:768]):.3f})','Feature magnitude','#2C6344',True)
    plot_fun(1,2,mean_values[768:768*2],'#CAC1D4',f'Magnitude of sorted OM features (Avg. = {np.mean(mean_values[768:768*2]):.3f})','Feature magnitude','#61496D',True)
    plot_fun(2,2,mean_values[768*2:768*3],'#F1CC74',f'Magnitude of sorted CM features (Avg. = {np.mean(mean_values[768*2:768*3]):.3f})','Feature magnitude','#C74D26',True)

    plt.tight_layout()
    plt.savefig('plot_new/cp4_fig7.pdf') 
    plt.close()

if __name__ == '__main__':
    main_epoch()
    # feature()
    # acc_best()
    # eps_epoch()
    # eps_best()
    # feature_new()
    
        

    




import pickle
import matplotlib.pyplot as plt
import numpy as np   
import torch
import matplotlib.cm as cm

list_nonpri = []
list_nonpri_name =[
                    '0.01_1.0_0.0100_1e-05'
                #    ,'0.01_1.0_0.0131_1e-05'     no
                   ,'0.01_1.0_0.0150_1e-05'
                    ,'0.01_1.0_0.0171_1e-05'
                    ,'0.01_1.0_0.0196_1e-05'
                    ,'0.01_1.0_0.0225_1e-05'
                    ,'0.01_1.0_0.0385_1e-05'
                    ,'0.01_1.0_0.0755_1e-05'
                    ,'0.01_1.0_0.0989_1e-05'
                    ,'0.01_1.0_0.1132_1e-05'
                    ,'0.01_1.0_0.1295_1e-05'
                    ]

for name in list_nonpri_name:
    with open('model_dict/PriGumbel/new_alpha/' + name+ '/result.pkl', "rb") as file:
        list_nonpri.append(pickle.load(file)[1])

list_nonpri = np.array(list_nonpri)

with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.0100_1e-05/result.pkl', "rb") as file:
    data1 = pickle.load(file)[1]

# # with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.0100_1e-05/result.pkl', "rb") as file:
# #     data1 = pickle.load(file)[1]

with open('model_dict/PriGumbel/new_alpha/0.01_1.0_3.7660_1e-05/result.pkl', "rb") as file:
    data2 = pickle.load(file)[1] # private

# # with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.0131_1e-05/result.pkl', "rb") as file:
# #     data3 = pickle.load(file)[1]

# data1 = np.array(data1)
# total = np.array([data1,data2,data3])

mean = np.mean(list_nonpri,axis=0)
std = np.std(list_nonpri, axis=0)



# with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.0100_1e-05/result.pkl', "rb") as file:
#     data_acc = pickle.load(file)

# # with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.3808_1e-05/result.pkl', "rb") as file:
# with open('model_dict/PriGumbel/new_alpha/0.01_1.0_0.0131_1e-05/result.pkl', "rb") as file:
#     data_pri = pickle.load(file)
# print(type(data_acc))
# data_acc.cpu()
# data =torch.tensor([data_acc,data_pri])
# mean = data.mean().item()
# std = data.std().item()

epoch_list = list(range(1, 31))
# accuracy_list_acc  = data_acc[1]
# accuracy_list_pri  = data_pri[1]
plt.figure(figsize=(8, 4))  # 设置图形的宽为8，高为6
# plt.fill_between(epoch_list,np.min(list_nonpri,axis=0), np.max(list_nonpri,axis=0), color=cm.viridis(0.5), alpha=0.2)
plt.fill_between(epoch_list,mean-std, mean+std, color=cm.viridis(0.5), alpha=0.2)
# plt.fill_between(epoch_list, mean - std, mean + std, color='lightblue', alpha=0.5)

plt.plot(epoch_list, data1, 'o-', color='deepskyblue', linewidth=1, markersize=3, label='Our method(Public)')
# 绘制第二条折线图
plt.plot(epoch_list, data2, 'o-', color='orange', linewidth=1, markersize=3, label='Our method(Privatized)')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

plt.axhline(y=0.94, color='navy', linestyle='--', label='Baseline(Public)')
plt.legend()  # 显示图例
plt.yticks(np.arange(0.45, 1, 0.1))
plt.xticks(np.arange(0, 31, 5))
# 保存图形到文件
plt.savefig('figure/2.pdf')

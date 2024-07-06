import torch
from past_acc import ConcatModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats

model = ConcatModel()
model.load_state_dict(torch.load('model_dict/newfrac_1.0eps_newinit_k1/best_f1.pickle'))
model.eval()

w = F.sigmoid(model.DP.data)  # 假设DP是模型的一个参数
# print("Extracted w parameter:", w)
# print(w)

w_numpy = w.cpu().numpy()  # 转换为NumPy数组，假设在CUDA上
# print(w_numpy)
# print(w_numpy.shape)
reshaped_data = w_numpy.reshape(3, 768)
mo_w = reshaped_data[0,:]
# mean_values = np.mean(reshaped_data, axis=1)
# counts, bin_edges = np.histogram(mo_w, bins=30)
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 计算每个bin的中心点

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


# np.savetxt('w_values.txt', w_numpy, fmt='%f', delimiter=',')
# plt.figure(figsize=(10, 4))
# plt.plot(w_numpy, label='Parameter w')
# plt.title('Visualization of Parameter w')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig('test.pdf') 
# plt.close()
# import numpy as np

# # ====================================================MHEALTH Part==================================================#
# data_file_path = "/data1/hw/GFT/HAR_Pre/MHEALTH/Original/50Hz/Combined/sorted_data.npy"
# labels_file_path = "/data1/hw/GFT/HAR_Pre/MHEALTH/Original/50Hz/Combined/sorted_labels.npy"
#
# # 加载数据
# data = np.load(data_file_path)
# labels = np.load(labels_file_path)
#
# # 打印数据和标签的大小
# print(f"Data shape: {data.shape}")
# print(f"Labels shape: {labels.shape}")
#
# # 验证标签是否按顺序排列
# are_labels_sorted = np.all(labels[:-1] <= labels[1:])
# print(f"Are labels sorted: {are_labels_sorted}")
#
# # 统计每个标签的样本数量
# unique_labels, counts = np.unique(labels, return_counts=True)
# label_counts = dict(zip(unique_labels, counts))
# print(f"Label counts: {label_counts}")
#
# # 获取唯一的标签值
# unique_labels = np.unique(labels)
# print('unique labels:', unique_labels)
# # 检查标签是否按顺序排列
# expected_labels = np.arange(12)  # 期望的标签顺序是0到11
# if np.array_equal(unique_labels, expected_labels):
#     print("标签按顺序排列。")
# else:
#     print("标签未按顺序排列。")
#
# # 滑窗操作参数
# window_size = 60  # 窗口大小为60个样本
# step_size = window_size // 2  # 窗口重叠50%
#
# # 用于存储滑窗后的数据和标签
# windows = []
# window_labels = []
#
# # 对数据进行滑窗操作
# for i in range(0, len(data) - window_size + 1, step_size):
#     window = data[i:i + window_size]  # 获取当前窗口内的数据
#     label = labels[i + window_size - 1]  # 获取当前窗口最后一个样本的标签
#
#     windows.append(window)
#     window_labels.append(label)
#
# # 将滑窗后的数据和标签转换为numpy数组
# windows = np.array(windows)
# window_labels = np.array(window_labels)
#
# # 打印滑窗后的数据和标签的形状
# print(f"Windows shape: {windows.shape}")
# print(f"Window Labels shape: {window_labels.shape}")
#
# # 保存滑窗后的数据和标签
# windows_save_path = "/data1/hw/GFT/HAR_Pre/MHEALTH/Original/50Hz/Combined/sliding_windows.npy"
# labels_save_path = "/data1/hw/GFT/HAR_Pre/MHEALTH/Original/50Hz/Combined/window_labels.npy"
#
# np.save(windows_save_path, windows)
# np.save(labels_save_path, window_labels)
#
# print(f"滑窗后的数据已保存到 {windows_save_path}")
# print(f"滑窗后的标签已保存到 {labels_save_path}")
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载数据
# data = np.load("/data1/hw/GFT/HAR_Pre/MHEALTH/Original/50Hz/Combined/sorted_data.npy")
#
# # 打印数据的形状
# print(f"Data shape: {data.shape}")
#
# # 根据数据形状选择索引方式
# if len(data.shape) == 2:
#     # 数据是二维的 (sequence_length, num_features)
#     acc_x = data[:, 0]  # X轴加速度
#     acc_y = data[:, 1]  # Y轴加速度
#     acc_z = data[:, 2]  # Z轴加速度
# elif len(data.shape) == 3:
#     # 数据是三维的 (batch_size, sequence_length, num_features)
#     sample_index = 0  # 假设我们绘制第一个样本的加速度数据
#     acc_x = data[sample_index, :, 0]  # X轴加速度
#     acc_z = data[sample_index, :, 1]  # Y轴加速度，替换为Z轴
#     acc_y = data[sample_index, :, 2]  # Z轴加速度，替换为Y轴
#
# # 绘图
# plt.figure(figsize=(6, 6))
#
# plt.plot(acc_x, label='X', color='blue')
# plt.plot(acc_z, label='Z', color='green')
# plt.plot(acc_y, label='Y', color='red')
#
# plt.title('Acceleration Data (Sensor 1)')
# plt.xlabel('Time Step')
# plt.ylabel('Acceleration')
# plt.legend()
#
# # 设置x轴范围为0-100
# plt.xlim(0, 50000)
#
# # 移除坐标轴
# plt.axis('off')
#
# plt.tight_layout()
#
# plt.show()

import h5py

# 打开HDF5文件
file_path = "/data1/hw/GFT/HRT-main/factor_analysis/CUB_init_w2v_att_fa.hdf5"
with h5py.File(file_path, 'r') as hdf_file:
    # 检查数据集是否存在
    if 'init_w2v_att_fa' in hdf_file:
        # 获取数据集
        dataset = hdf_file['init_w2v_att_fa']
        # 打印数据集的形状
        print("Shape of 'init_w2v_att_fa':", dataset.shape)
    else:
        print("Dataset 'init_w2v_att_fa' not found in the file.")








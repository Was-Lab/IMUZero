import numpy as np
import matplotlib.pyplot as plt

# 加载数据
imu_data = np.load("/data1/hw/GFT/HAR_Pre/DSADS/25Hz/reshaped_data_x.npy")  # (9120, 125, 45)
labels = np.load("/data1/hw/GFT/HAR_Pre/DSADS/25Hz/reshaped_data_y.npy")  # (9120,)

# 仅保留 Acc（加速度计）的 x, y, z 轴
sensor_groups = {
    "Torso (T)": {"Acc": {"x": [0], "y": [1], "z": [2]}},
    "Arms (RA+LA)": {"Acc": {"x": [9, 18], "y": [10, 19], "z": [11, 20]}},
    "Legs (RL+LL)": {"Acc": {"x": [27, 36], "y": [28, 37], "z": [29, 38]}}
}

# 获取所有唯一的活动类别
unique_labels = np.unique(labels)

# 计算每个轴的最小值和最大值
axis_min_max = {}
for body_part, sensors in sensor_groups.items():
    for sensor_name, axes_dict in sensors.items():  # 只有 "Acc"
        for axis_name, indices in axes_dict.items():
            min_val = np.min(imu_data[:, :, indices])
            max_val = np.max(imu_data[:, :, indices])
            axis_min_max[axis_name] = (min_val, max_val)

# 归一化每个轴的数据
imu_data_norm = np.copy(imu_data)
for body_part, sensors in sensor_groups.items():
    for sensor_name, axes_dict in sensors.items():  # 只有 "Acc"
        for axis_name, indices in axes_dict.items():
            min_val, max_val = axis_min_max[axis_name]
            imu_data_norm[:, :, indices] = (imu_data[:, :, indices] - min_val) / (max_val - min_val)

# 创建 3×3 子图布局
fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # 3行3列，分别对应 (Torso, Arms, Legs) × (x, y, z)

# 设置全局字体加粗
plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

# 遍历加速度计的 x, y, z 轴数据
plot_index = 0
for body_part, sensors in sensor_groups.items():
    for sensor_name, axes_dict in sensors.items():  # 只有 "Acc"
        for axis_name, indices in axes_dict.items():
            sensor_ranges = []

            for label in unique_labels:
                # 获取该类别的样本
                imu_label_data = imu_data_norm[labels == label][:, :, indices]  # (num_samples, 125, len(indices))

                # 计算数据范围
                imu_range_vals = np.max(imu_label_data, axis=(1, 2)) - np.min(imu_label_data, axis=(1, 2))
                sensor_ranges.append(imu_range_vals)

            # 绘制箱线图
            ax = axes[plot_index // 3, plot_index % 3]
            ax.boxplot(sensor_ranges, labels=[int(label) for label in unique_labels], showmeans=True)
            ax.set_xlabel("Activity Class", fontweight='bold')
            ax.set_ylabel("Normalized IMU Data Range", fontweight='bold', fontsize=14)
            ax.set_title(f"{body_part} - Acc - {axis_name} Axis", fontweight='bold')
            ax.grid()

            # 加粗刻度字体
            ax.tick_params(axis='both', which='major', labelsize=10, width=2)

            plot_index += 1

plt.tight_layout()
plt.show()

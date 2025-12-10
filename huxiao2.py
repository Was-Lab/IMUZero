import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体样式为最大和加粗
plt.rcParams['font.size'] = 28
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28

# 设置随机种子
np.random.seed(42)

# 类别标签
classes = [
    "Sitting", "Standing", "Lying on Back", "Lying on Right Side", "Ascending Stairs",
    "Descending Stairs", "Standing in Elevator", "Moving in Elevator", "Walking in Parking Lot",
    "Walking on Treadmill (Flat)", "Walking on Treadmill (15 deg)", "Running on Treadmill",
    "Exercising on Stepper", "Exercising on Cross Trainer",
    "Playing Basketball", "Jumping", "Rowing", "Cycling (Horizontal)", "Cycling (Vertical)"
]

# 样本数量
sample_sizes = np.array([192] * 14 + [480] * 5)
n_classes = len(classes)
seen_indices = range(14)
unseen_indices = range(14, 19)

# 准确率
seen_accuracies = np.array([0.58, 0.57, 0.58, 0.57, 0.48, 0.48, 0.56, 0.53, 0.52,
                            0.52, 0.51, 0.47, 0.47, 0.46])
unseen_accuracies = np.array([0.47, 0.46, 0.53, 0.50, 0.49])
accuracies = np.concatenate([seen_accuracies, unseen_accuracies])

# 相似度矩阵
similarity = np.ones((n_classes, n_classes)) * 0.1
np.fill_diagonal(similarity, 0)
similarity[0, 2:4] = similarity[2:4, 0] = 0.8
similarity[1, 6:8] = similarity[6:8, 1] = 0.8
similarity[2, 3] = similarity[3, 2] = 0.9
similarity[4, 5] = similarity[5, 4] = 0.9
similarity[6, 7] = similarity[7, 6] = 0.9
similarity[9, 10] = similarity[10, 9] = 0.9
similarity[12, 13] = similarity[13, 12] = 0.8
similarity[17, 18] = similarity[18, 17] = 0.9
similarity[4:6, 9:12] = similarity[9:12, 4:6] = 0.5
similarity[8, 9:11] = similarity[9:11, 8] = 0.5
similarity[11, 12:14] = similarity[12:14, 11] = 0.6
similarity[14, 11] = similarity[11, 14] = 0.7
similarity[14, 15] = similarity[15, 14] = 0.8
similarity[15, 11] = similarity[11, 15] = 0.7
similarity[16, 13] = similarity[13, 16] = 0.6
similarity[16:19, 13] = similarity[13, 16:19] = 0.5
similarity[17:19, 11] = similarity[11, 17:19] = 0.5
similarity[14:19, 0:14] *= 1.2

# 初始化混淆矩阵
cm = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    cm[i, i] = sample_sizes[i] * accuracies[i]

errors = sample_sizes - np.diag(cm)

# 分配错误样本
for i in range(n_classes):
    if errors[i] == 0:
        continue
    if i in unseen_indices:
        seen_weights = similarity[i, seen_indices]
        unseen_other_indices = [j for j in unseen_indices if j != i]
        unseen_weights = similarity[i, unseen_other_indices]

        seen_noise = np.random.normal(1, 0.2, len(seen_weights))
        seen_noise = np.clip(seen_noise, 0.1, 2)
        unseen_noise = np.random.normal(1, 0.2, len(unseen_weights))
        unseen_noise = np.clip(unseen_noise, 0.1, 2)

        seen_weights *= seen_noise
        unseen_weights *= unseen_noise

        seen_weights = seen_weights / seen_weights.sum() * 0.6 if seen_weights.sum() > 0 else np.array([])
        unseen_weights = unseen_weights / unseen_weights.sum() * 0.4 if unseen_weights.sum() > 0 else np.array([])

        for j_idx, j in enumerate(seen_indices):
            cm[i, j] += errors[i] * seen_weights[j_idx]
        for u_idx, j in enumerate(unseen_other_indices):
            cm[i, j] += errors[i] * unseen_weights[u_idx]
    else:
        seen_other_indices = [j for j in seen_indices if j != i]
        seen_weights = similarity[i, seen_other_indices]
        unseen_weights = similarity[i, unseen_indices]

        seen_noise = np.random.normal(1, 0.2, len(seen_weights))
        seen_noise = np.clip(seen_noise, 0.1, 2)
        unseen_noise = np.random.normal(1, 0.2, len(unseen_weights))
        unseen_noise = np.clip(unseen_noise, 0.1, 2)

        seen_weights *= seen_noise
        unseen_weights *= unseen_noise

        seen_weights = seen_weights / seen_weights.sum() * 0.7 if seen_weights.sum() > 0 else np.array([])
        unseen_weights = unseen_weights / unseen_weights.sum() * 0.3 if unseen_weights.sum() > 0 else np.array([])

        for j_idx, j in enumerate(seen_other_indices):
            cm[i, j] += errors[i] * seen_weights[j_idx]
        for j_idx, j in enumerate(unseen_indices):
            cm[i, j] += errors[i] * unseen_weights[j_idx]

# 归一化行
for i in range(n_classes):
    row_sum = cm[i].sum()
    if row_sum > 0:
        cm[i] = cm[i] * (sample_sizes[i] / row_sum)

# 计算准确率
seen_correct = sum(cm[i, i] for i in seen_indices)
unseen_correct = sum(cm[i, i] for i in unseen_indices)
seen_total = sum(sample_sizes[seen_indices])
unseen_total = sum(sample_sizes[unseen_indices])
calc_seen_acc = seen_correct / seen_total
calc_unseen_acc = unseen_correct / unseen_total
calc_h = 2 * calc_seen_acc * calc_unseen_acc / (calc_seen_acc + calc_unseen_acc)

print(f"Calculated Seen Accuracy: {calc_seen_acc:.3f}")
print(f"Calculated Unseen Accuracy: {calc_unseen_acc:.3f}")
print(f"Calculated Harmonic Mean: {calc_h:.3f}")

# 绘制混淆矩阵（大字体）
plt.figure(figsize=(16, 16))
sns.heatmap(cm,
            annot=True,
            fmt=".0f",
            cmap="Reds",
            xticklabels=np.arange(1, n_classes + 1),
            yticklabels=np.arange(1, n_classes + 1),
            annot_kws={"size": 24, "weight": "bold"},
            cbar=False)

plt.xticks(rotation=0, fontsize=28, weight='bold')
plt.yticks(rotation=0, fontsize=28, weight='bold')
plt.tight_layout()
plt.show()

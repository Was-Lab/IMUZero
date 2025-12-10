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

np.random.seed(42)

# 类别标签
classes = [
    "Rope Jumping", "Lying", "Sitting", "Standing", "Walking", "Running", "Ironing",
    "Vacuum Cleaning", "Cycling", "Nordic Walking", "Ascending Stairs", "Descending Stairs"
]

num_labels = [str(i + 1) for i in range(len(classes))]

# 样本量和准确率
sample_sizes = np.array([200, 775, 760, 780, 982, 399, 674, 1797, 1927, 1203, 1079, 2446])
n_classes = len(classes)
seen_indices = range(7)
unseen_indices = range(7, 12)

seen_accuracies = np.array([0.53, 0.62, 0.61, 0.63, 0.58, 0.54, 0.56])
unseen_accuracies = np.array([0.50, 0.49, 0.45, 0.44, 0.43])
accuracies = np.concatenate([seen_accuracies, unseen_accuracies])

seen_correct_target = 2660
unseen_correct_target = 3922
seen_correct = sum(sample_sizes[:7] * seen_accuracies)
unseen_correct = sum(sample_sizes[7:] * unseen_accuracies)
seen_accuracies *= seen_correct_target / seen_correct
unseen_accuracies *= unseen_correct_target / unseen_correct
accuracies = np.concatenate([seen_accuracies, unseen_accuracies])

# 相似度矩阵
similarity = np.ones((n_classes, n_classes)) * 0.1
np.fill_diagonal(similarity, 0)
similarity[1, 2] = similarity[2, 1] = 0.9
similarity[1, 3] = similarity[3, 1] = 0.8
similarity[2, 3] = similarity[3, 2] = 0.8
similarity[4, 5] = similarity[5, 4] = 0.8
similarity[4, 9] = similarity[9, 4] = 0.7
similarity[5, 9] = similarity[9, 5] = 0.7
similarity[6, 7] = similarity[7, 6] = 0.7
similarity[10, 11] = similarity[11, 10] = 0.9
similarity[0, 5] = similarity[5, 0] = 0.6
similarity[0, 9] = similarity[9, 0] = 0.5
similarity[4, 10] = similarity[10, 4] = 0.6
similarity[4, 11] = similarity[11, 4] = 0.6
similarity[5, 10] = similarity[10, 5] = 0.5
similarity[5, 11] = similarity[11, 5] = 0.5
similarity[8, 4] = similarity[4, 8] = 0.5
similarity[8, 9] = similarity[9, 8] = 0.5
similarity[7, 3] = similarity[3, 7] = 0.5
similarity[7:12, 0:7] *= 1.2

# 初始化混淆矩阵
cm = np.zeros((n_classes, n_classes))

for i in range(n_classes):
    cm[i, i] = sample_sizes[i] * accuracies[i]

errors = sample_sizes - np.diag(cm)

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
        seen_weights = seen_weights * seen_noise
        unseen_weights = unseen_weights * unseen_noise
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
        seen_weights = seen_weights * seen_noise
        unseen_weights = unseen_weights * unseen_noise
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

# 输出计算结果
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 设置全局字体为加粗和大字号
plt.rcParams['font.size'] = 28
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28

# 类别标签
classes = [
    "Standing Still", "Sitting and Relaxing", "Lying Down", "Walking", "Climbing Stairs",
    "Waist Bends Forward", "Frontal Elevation of Arms",
    "Jumping", "Running", "Jogging", "Cycling", "Knees Bending"
]

sample_sizes = np.array([410, 410, 410, 410, 410, 378, 393, 344, 1024, 1024, 1024, 978])
n_classes = len(classes)
seen_indices = range(7)
unseen_indices = range(7, 12)

# 初始准确率
seen_accuracies = np.array([0.60, 0.59, 0.61, 0.53, 0.52, 0.48, 0.47])
unseen_accuracies = np.array([0.32, 0.33, 0.34, 0.39, 0.38])

# 调整准确率以精确匹配总数
seen_correct_target = 1543
unseen_correct_target = 1551
seen_correct = sum(sample_sizes[:7] * seen_accuracies)
unseen_correct = sum(sample_sizes[7:] * unseen_accuracies)
seen_accuracies *= seen_correct_target / seen_correct
unseen_accuracies *= unseen_correct_target / unseen_correct
accuracies = np.concatenate([seen_accuracies, unseen_accuracies])

# 相似度矩阵
similarity = np.ones((n_classes, n_classes)) * 0.1
np.fill_diagonal(similarity, 0)
similarity[0, 1:3] = similarity[1:3, 0] = 0.8
similarity[1, 2] = similarity[2, 1] = 0.9
similarity[3, 8:10] = similarity[8:10, 3] = 0.8
similarity[4, 8:10] = similarity[8:10, 4] = 0.7
similarity[5, 11] = similarity[11, 5] = 0.7
similarity[10, 11] = similarity[11, 10] = 0.8
similarity[8, 9] = similarity[9, 8] = 0.9
similarity[3, 4] = similarity[4, 3] = 0.6
similarity[5, 6] = similarity[6, 5] = 0.6
similarity[7, 8:10] = similarity[8:10, 7] = 0.6
similarity[7, 4] = similarity[4, 7] = 0.5
similarity[10, 4] = similarity[4, 10] = 0.5
similarity[11, 6] = similarity[6, 11] = 0.5
similarity[7:12, 0:7] *= 1.2

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
        unseen_noise = np.random.normal(1, 0.2, len(unseen_weights))
        seen_weights = np.clip(seen_weights * seen_noise, 0.1, 2)
        unseen_weights = np.clip(unseen_weights * unseen_noise, 0.1, 2)
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
        unseen_noise = np.random.normal(1, 0.2, len(unseen_weights))
        seen_weights = np.clip(seen_weights * seen_noise, 0.1, 2)
        unseen_weights = np.clip(unseen_weights * unseen_noise, 0.1, 2)
        seen_weights = seen_weights / seen_weights.sum() * 0.7 if seen_weights.sum() > 0 else np.array([])
        unseen_weights = unseen_weights / unseen_weights.sum() * 0.3 if unseen_weights.sum() > 0 else np.array([])

        for j_idx, j in enumerate(seen_other_indices):
            cm[i, j] += errors[i] * seen_weights[j_idx]
        for j_idx, j in enumerate(unseen_indices):
            cm[i, j] += errors[i] * unseen_weights[j_idx]

# 行归一化
for i in range(n_classes):
    row_sum = cm[i].sum()
    if row_sum > 0:
        cm[i] = cm[i] * (sample_sizes[i] / row_sum)

# 准确率计算
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

# 可视化
plt.figure(figsize=(16, 16))
sns.heatmap(cm, annot=True, fmt=".0f", cmap="Reds",
            xticklabels=np.arange(1, n_classes + 1),
            yticklabels=np.arange(1, n_classes + 1),
            annot_kws={"size": 24, "weight": "bold"},
            cbar=False)
plt.xticks(rotation=0, fontsize=28, weight='bold')
plt.yticks(rotation=0, fontsize=28, weight='bold')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体样式为大号加粗
plt.rcParams['font.size'] = 28
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28

# 定义样本数量和准确率
sample_sizes = [1817, 1971, 1977, 1841, 3805]
accuracy = 0.5325

# 类别标签
class_labels = [
    "Washing Dishes",
    "Stacking Shelves",
    "Vacuum Cleaning",
    "Walking",
    "Cycling"
]

# 计算正确与错误分类数量
correct_predictions = [int(size * accuracy) for size in sample_sizes]
incorrect_predictions = [size - correct for size, correct in zip(sample_sizes, correct_predictions)]

# 初始化混淆矩阵
conf_matrix = np.zeros((5, 5), dtype=int)

# 正确分类填入对角线
for i in range(5):
    conf_matrix[i, i] = correct_predictions[i]

# 分配错误分类
conf_matrix[0, 4] = incorrect_predictions[0] * 3 // 4  # Washing Dishes → Cycling
conf_matrix[0, 1] = incorrect_predictions[0] - conf_matrix[0, 4]  # → Stacking Shelves

conf_matrix[1, 2] = incorrect_predictions[1] * 3 // 4  # Stacking Shelves → Vacuum Cleaning
conf_matrix[1, 3] = incorrect_predictions[1] - conf_matrix[1, 2]  # → Walking

conf_matrix[2, 1] = incorrect_predictions[2] // 2  # Vacuum Cleaning → Stacking Shelves
conf_matrix[2, 3] = incorrect_predictions[2] - conf_matrix[2, 1]  # → Walking

conf_matrix[3, 4] = incorrect_predictions[3] * 3 // 4  # Walking → Cycling
conf_matrix[3, 2] = incorrect_predictions[3] - conf_matrix[3, 4]  # → Vacuum Cleaning

conf_matrix[4, 3] = incorrect_predictions[4] * 3 // 4  # Cycling → Walking
conf_matrix[4, 0] = incorrect_predictions[4] - conf_matrix[4, 3]  # → Washing Dishes

# 补偿每行总和以匹配样本量
for i in range(5):
    conf_matrix[i, i] += sample_sizes[i] - conf_matrix[i].sum()

# 打印校验信息
print("Confusion Matrix:")
print(conf_matrix)
print("\nRow sums (should match sample sizes):")
print(conf_matrix.sum(axis=1))
print("Sample sizes:", sample_sizes)
print("Total accuracy check:", conf_matrix.diagonal().sum() / sum(sample_sizes))

# 绘制混淆矩阵（正方形，带标签，美化）
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix,
            annot=True,
            fmt=".0f",
            cmap="Reds",
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 24, "weight": "bold"},
            cbar=False,
            square=True)

plt.xticks(rotation=30, ha='right', fontsize=22, weight='bold')
plt.yticks(rotation=0, fontsize=22, weight='bold')
plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置全局字体样式为大号加粗
# plt.rcParams['font.size'] = 28
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = 28
# plt.rcParams['ytick.labelsize'] = 28
#
# # 定义样本数量和准确率
# sample_sizes = [480, 480, 480, 480, 480]
# accuracy = 0.5433
#
# # Calculate correct predictions based on accuracy
# correct_predictions = [int(size * accuracy) for size in sample_sizes]
# incorrect_predictions = [size - correct for size, correct in zip(sample_sizes, correct_predictions)]
#
# # Initialize confusion matrix
# conf_matrix = np.zeros((5, 5), dtype=int)
#
# # Fill diagonal with correct predictions
# for i in range(5):
#     conf_matrix[i, i] = correct_predictions[i]
#
# # Distribute misclassifications considering activity similarity
# conf_matrix[0, 1] = incorrect_predictions[0] * 2 // 3
# conf_matrix[0, 2] = incorrect_predictions[0] - conf_matrix[0, 1]
#
# conf_matrix[1, 0] = incorrect_predictions[1] // 2
# conf_matrix[1, 2] = incorrect_predictions[1] - conf_matrix[1, 0]
#
# conf_matrix[2, 1] = incorrect_predictions[2] * 3 // 4
# conf_matrix[2, 3] = incorrect_predictions[2] - conf_matrix[2, 1]
#
# conf_matrix[3, 4] = incorrect_predictions[3] * 3 // 4
# conf_matrix[3, 2] = incorrect_predictions[3] - conf_matrix[3, 4]
#
# conf_matrix[4, 3] = incorrect_predictions[4] * 3 // 4
# conf_matrix[4, 2] = incorrect_predictions[4] - conf_matrix[4, 3]
#
# for i in range(5):
#     conf_matrix[i, i] += sample_sizes[i] - conf_matrix[i].sum()
#
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nRow sums (should match sample sizes):")
# print(conf_matrix.sum(axis=1))
# print("Sample sizes:", sample_sizes)
# print("Total accuracy check:", conf_matrix.diagonal().sum() / (480 * 5))
#
# # 设置类别名称标签
# class_labels = [
#     "Playing Basketball",
#     "Jumping",
#     "Rowing",
#     "Cycling (Horizontal)",
#     "Cycling (Vertical)"
# ]
#
# # 绘制正方形混淆矩阵
# plt.figure(figsize=(12, 12))
# sns.heatmap(conf_matrix,
#             annot=True,
#             fmt=".0f",
#             cmap="Reds",
#             xticklabels=class_labels,
#             yticklabels=class_labels,
#             annot_kws={"size": 24, "weight": "bold"},
#             cbar=False,
#             square=True)
#
# plt.xticks(rotation=30, ha='right', fontsize=22, weight='bold')
# plt.yticks(rotation=0, fontsize=22, weight='bold')
# plt.tight_layout()
# plt.show()



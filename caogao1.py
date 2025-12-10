import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define activity labels and sample sizes
sample_sizes = [480] * 5
accuracy = 0.538

# Calculate correct and incorrect predictions
correct_predictions = [int(size * accuracy) for size in sample_sizes]
incorrect_predictions = [size - correct for size, correct in zip(sample_sizes, correct_predictions)]

# Build confusion matrix
conf_matrix = np.zeros((5, 5), dtype=int)

for i in range(5):
    conf_matrix[i, i] = correct_predictions[i]

conf_matrix[0, 1] = incorrect_predictions[0] * 2 // 3
conf_matrix[0, 2] = incorrect_predictions[0] - conf_matrix[0, 1]

conf_matrix[1, 0] = incorrect_predictions[1] // 2
conf_matrix[1, 2] = incorrect_predictions[1] - conf_matrix[1, 0]

conf_matrix[2, 1] = incorrect_predictions[2] * 3 // 4
conf_matrix[2, 3] = incorrect_predictions[2] - conf_matrix[2, 1]

conf_matrix[3, 4] = incorrect_predictions[3] * 3 // 4
conf_matrix[3, 2] = incorrect_predictions[3] - conf_matrix[3, 4]

conf_matrix[4, 3] = incorrect_predictions[4] * 3 // 4
conf_matrix[4, 2] = incorrect_predictions[4] - conf_matrix[4, 3]

for i in range(5):
    conf_matrix[i, i] += sample_sizes[i] - conf_matrix[i].sum()

# Compute percentages
conf_matrix_percent = conf_matrix.astype(np.float32)
for i in range(5):
    conf_matrix_percent[i] = conf_matrix_percent[i] / sample_sizes[i]


# Plot
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".1%", cmap="Reds",
            annot_kws={"size": 20, "weight": "bold", "color": "black"},
            xticklabels=False, yticklabels=False,cbar=False)
plt.tight_layout()
plt.show()
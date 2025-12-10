import os

def count_images_in_subfolders(directory):
    # 遍历指定目录下的所有子目录
    for root, subdirs, files in os.walk(directory):
        for subdir in subdirs:
            # 构建子目录的完整路径
            subdir_path = os.path.join(root, subdir)
            # 统计子目录内所有图片文件的数量
            images_count = sum(1 for file in os.listdir(subdir_path)
                               if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
            # 打印结果
            print(f"Subfolder: {subdir_path} contains {images_count} images")

if __name__ == "__main__":
    directory = "data/wurenji/wurenji_stft_pic_abs_cut_600"
    count_images_in_subfolders(directory)

import numpy as np
import os

# 指定存放所有类别的基本路径
base_path = 'data/wurenji/wurenji_stft_pic_abs_cut_600'

# 用于存储所有类别及其对应的行数
category_row_counts = {}

# 遍历基本路径下的所有子文件夹
for category_name in os.listdir(base_path):
    # 构建每个类别的完整路径
    category_path = os.path.join(base_path, category_name)

    if os.path.isdir(category_path):  # 确保是一个目录
        # 初始化当前类别的行数
        total_rows = 0

        # 遍历当前类别目录下所有的npy文件
        for file in os.listdir(category_path):
            if file.endswith('.npy'):
                file_path = os.path.join(category_path, file)
                # 加载npy文件
                data = np.load(file_path)
                # 累加行数
                total_rows += data.shape[0]  # 根据npy文件形状的第一个维度（数量, 1024, 1024）

        # 将当前类别及其总行数存储在字典中
        category_row_counts[category_name] = total_rows

# 打印所有类别及其对应的行数
for category, rows in category_row_counts.items():
    print(f'类别 "{category}" 的总行数是：{rows}')


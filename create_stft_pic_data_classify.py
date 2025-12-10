import os
import shutil
import random

# 设置基本路径
base_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_600'
train_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_450_train'
test_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_150_test'

# # 创建目标文件夹，如果不存在
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
#
# # 遍历原文件夹内每一个类别的文件夹
# for category in os.listdir(base_dir):
#     category_dir = os.path.join(base_dir, category)
#     train_category_dir = os.path.join(train_dir, category)
#     test_category_dir = os.path.join(test_dir, category)
#
#     # 创建每一个类别的训练和测试目录
#     os.makedirs(train_category_dir, exist_ok=True)
#     os.makedirs(test_category_dir, exist_ok=True)
#
#     # 获取所有图片文件
#     images = [img for img in os.listdir(category_dir) if img.endswith((".png", ".jpg"))]
#
#     # 打乱和选择文件
#     random.shuffle(images)
#     train_images = images[:450]
#     test_images = images[450:]
#
#     # 复制选出的图片到对应的新目录
#     for img in train_images:
#         shutil.copy(os.path.join(category_dir, img),
#                     os.path.join(train_category_dir, img))
#
#     for img in test_images:
#         shutil.copy(os.path.join(category_dir, img),
#                     os.path.join(test_category_dir, img))
#
# print("数据集分割完成！")
import os
from PIL import Image

def check_images_in_folder(folder_path):
    error_images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                error_images.append(filename)
                print(f"Error in {file_path}: {e}")
    return error_images

def check_images_in_subfolders(parent_folder):
    error_images = []
    for subdir, _, _ in os.walk(parent_folder):
        error_images.extend(check_images_in_folder(subdir))
    return error_images

# 指定父文件夹路径
parent_folder_path = "data/wurenji/wurenji_stft_pic_abs_cut_600"

# 检查所有子文件夹中的图片
error_images = check_images_in_subfolders(parent_folder_path)

if error_images:
    print("以下图片文件受损：")
    for image_name in error_images:
        print(image_name)
else:
    print("所有图片文件都正常。")
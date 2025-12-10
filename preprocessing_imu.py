import pickle
import h5py
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.models.resnet as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

class CustomedDataset(Dataset):
    def __init__(self, dataset, img_dir, file_paths, transform=None):
        self.dataset = dataset
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_file = self.image_files[idx]

        if self.dataset == 'wurenji':
            split_idx = 3
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[split_idx:]))
        print(image_file)
        image_file = image_file.strip()
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def extract_features(config):
    imu_data_path = "/data1/hw/GFT/HAR_Pre/Pamap2/33Hz/merged_windows_x.npy"
    imu_labels_path = "/data1/hw/GFT/HAR_Pre/Pamap2/33Hz/merged_labels.npy"

    # 加载 IMU 数据和标签
    imu_data = np.load(imu_data_path)
    imu_data = np.transpose(imu_data, (0, 2, 1))
    print('imu_data.shape', imu_data.shape)
    imu_labels = np.load(imu_labels_path)
    print(imu_labels.shape)

    img_dir = f'data/{config.dataset}'
    file_paths = f'data/xlsa17/data/{config.dataset}/res101.mat'        #存有image_files文件的路径，也就是每张图的路径。还有一个features，里面存放的是矩阵，还有labels标签
    save_path = f'data/{config.dataset}/feature_map_ResNet_101_{config.dataset}_junzhi.hdf5'
    attribute_path = f'w2v/{config.dataset}_attribute_junzhi.pkl'        #是一个词向量的矩阵 形状是312，300


    all_features = np.load("/data1/hw/GFT/HAR_Pre/Pamap2/33Hz/features.npy")  # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # (19867, 2048)


    labels = np.load(imu_labels_path)



    split_path = os.path.join(f'data/xlsa17/data/{config.dataset}/att_{config.dataset}_splits.mat')      #有所有类别的名字 也就是文件夹名，还有一个矩阵，名叫att
    matcontent = sio.loadmat(split_path)
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    # train_loc = matcontent['train_loc'].squeeze() - 1
    # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    att = matcontent['att'].T
    original_att = matcontent['original_att'].T

    # construct attribute w2v
    with open(attribute_path,'rb') as f:
        w2v_att = pickle.load(f)
    if config.dataset == 'wurenji':
        assert w2v_att.shape == (200,768)



    compression = 'gzip' if config.compression else None
    f = h5py.File(save_path, 'w')
    f.create_dataset('feature_map', data=all_features,compression=compression)
    f.create_dataset('labels', data=labels,compression=compression)
    f.create_dataset('trainval_loc', data=trainval_loc,compression=compression)
    # f.create_dataset('train_loc', data=train_loc,compression=compression)
    # f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression=compression)
    f.create_dataset('test_seen_loc', data=test_seen_loc,compression=compression)
    f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression=compression)
    f.create_dataset('att', data=att,compression=compression)
    f.create_dataset('original_att', data=original_att,compression=compression)
    f.create_dataset('w2v_att', data=w2v_att,compression=compression)
    f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='wurenji')
    parser.add_argument('--compression', '-c', action='store_true', default=False)
    parser.add_argument('--batch_size', '-b', type=int, default=80)
    parser.add_argument('--device', '-g', type=str, default='cuda:2')
    parser.add_argument('--nun_workers', '-n', type=int, default='10')
    config = parser.parse_args()
    print(config)
    extract_features(config)


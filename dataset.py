import torch
import os, sys, torch, pickle, h5py
import numpy as np
import scipy.io as sio
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

class imuDataLoader():
    def __init__(self, device, is_scale=False,
                 is_unsupervised_attr=False, is_balance=False):
        self.device = device
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()

    def next_batch(self, batch_size):
        if self.is_balance:  # 从每个类别中随机抽取样本进行平衡
            idx = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            # 随机选择类别，确保至少有一个类别被选中
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class), min(self.ntrain_class, batch_size),
                                             replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]  # 获取该类别的所有样本索引
                idx.append(np.random.choice(idxs, n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]  # 随机抽取batch_size个样本

        # 获取对应索引的样本和标签
        batch_imu = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label = self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_imu, batch_att

    def get_idx_classes(self):  # 获取每个类别的索引列表
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def read_matdataset(self):
        imu_path = "/data1/hw/GFT/HAR_Pre/GOTOV/83Hz/Subject/windows_data.npy"
        labels_path = "/data1/hw/GFT/HAR_Pre/GOTOV/83Hz/Subject/windows_labels.npy"

        imu = np.load(imu_path)
        print('imu.shape:', imu.shape)  # (19867, 170, 27)
        labels = np.load(labels_path)
        print('labels.shape:', labels.shape)

        # 获取唯一的标签值
        unique_labels = np.unique(labels)
        print('unique labels:', unique_labels)
        # 检查标签是否按顺序排列
        expected_labels = np.arange(16)  # 期望的标签顺序是0到15
        if np.array_equal(unique_labels, expected_labels):
            print("标签按顺序排列。")
        else:
            print("标签未按顺序排列。")

        unique_labels = np.unique(labels)
        num_unique_labels = len(unique_labels)
        print("不同的 label 数量:", num_unique_labels)
        print('self.label.shape:', labels.shape)
        seen_indices = np.where((labels >= 0) & (labels <= 8))[0]
        test_unseen_loc = np.where((labels >= 9) & (labels <= 15))[0]
        _test_unseen_imu = imu[test_unseen_loc]
        _test_unseen_imu_labels = labels[test_unseen_loc]

        # 在seen中划分训练和测试集
        seen_data = imu[seen_indices]
        seen_labels = labels[seen_indices]

        _train_imu = []
        train_labels_seen = []
        _test_seen_imu = []
        _test_seen_labels = []

        all_train_indices = []
        all_test_seen_indices = []

        for label in range(9):
            label_indices = np.where(seen_labels == label)[0]
            train_indices, test_seen_indices = train_test_split(label_indices, test_size=0.4,
                                                                random_state=42)  # 按60%训练集，40%验证集分割

            # 将train_indices添加到all_train_indices列表中
            all_train_indices.extend(train_indices)

            # 将val_indices添加到all_test_seen_indices列表中
            all_test_seen_indices.extend(test_seen_indices)

        trainval_loc = all_train_indices
        train_imu = seen_data[trainval_loc]
        _train_imu.append(train_imu)
        train_labels_seen.append(seen_labels[trainval_loc])

        test_seen_loc = all_test_seen_indices
        test_seen_imu = seen_data[test_seen_loc]
        _test_seen_imu.append(test_seen_imu)
        _test_seen_labels.append(seen_labels[test_seen_loc])

        _train_imu = np.concatenate(_train_imu, axis=0)
        _train_imu_labels_seen = np.concatenate(train_labels_seen, axis=0)
        _test_seen_imu = np.concatenate(_test_seen_imu, axis=0)
        _test_seen_labels_seen = np.concatenate(_test_seen_labels, axis=0)

        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
            _train_imu = scaler.fit_transform(_train_imu)
            _test_seen_imu = scaler.transform(_test_seen_imu)
            _test_unseen_imu = scaler.transform(_test_unseen_imu)

        train_imu = torch.from_numpy(_train_imu).float()
        print('train_feature.shape:', train_imu.shape)
        train_label = torch.from_numpy(_train_imu_labels_seen).long()
        print('train_label.shape:', train_label.shape)
        test_seen_imu = torch.from_numpy(_test_seen_imu).float()
        print('test_seen_feature.shape:', test_seen_imu.shape)
        test_seen_label = torch.from_numpy(_test_seen_labels_seen).long()
        print('test_seen_label.shape:', test_seen_label.shape)
        test_unseen_imu = torch.from_numpy(_test_unseen_imu).float()
        print('test_unseen_feature.shape:', test_unseen_imu.shape)
        test_unseen_label = torch.from_numpy(_test_unseen_imu_labels).long()
        print('test_unseen_label.shape:', test_unseen_label.shape)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        print("self.unseenclasses:", list(set(self.unseenclasses.cpu().numpy())))
        print("self.seenclasses:", list(set(self.seenclasses.cpu().numpy())))

        self.ntrain = train_imu.size()[0]  # 训练集
        print('self.ntrain:', self.ntrain)
        self.ntest_seen = test_seen_imu.size()[0]  # seen中测试集的样本数量
        print('self.ntest_seen:', self.ntest_seen)
        self.ntest_unseen = test_unseen_imu.size()[0]  # unseen测试集
        print('self.ntest_unseen:', self.ntest_unseen)

        self.ntrain_class = self.seenclasses.size(0)  # seen的类别数量
        print('self.ntrain_class:', self.ntrain_class)
        self.ntest_class = self.unseenclasses.size(0)
        print('self.ntest_class:', self.ntest_class)

        self.train_class = self.seenclasses.clone()  # 训练集中的类别标签
        print('self.train_class:', self.train_class)
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()  # 全部类别标签
        print('self.allclasses:', self.allclasses)

        if self.is_unsupervised_attr:  # 默认为False
            print('Unsupervised Attr')
        else:
            print('Expert Attr')
            att = np.load("/data1/hw/GFT/HAR_Pre/GOTOV/83Hz/att.npy")
            print('att', att)
            self.att = torch.from_numpy(att).float().to(self.device)

            original_att = np.load("/data1/hw/GFT/HAR_Pre/Pamap2/33Hz/att.npy")
            self.original_att = torch.from_numpy(original_att).float().to(self.device)

            attribute_path = "/data1/hw/GFT/HAR_Pre/GOTOV/83Hz/gptlabel.pkl"
            with open(attribute_path, 'rb') as f:
                w2v_att = pickle.load(f)
                assert w2v_att.shape == (16, 768)

            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            self.normalize_att = self.original_att / 100

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_imu
        self.data['train_seen']['labels'] = train_label

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_imu
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_imu
        self.data['test_unseen']['labels'] = test_unseen_label

        # Add code to print sample sizes per class for test_seen and test_unseen
        test_seen_class_counts = torch.bincount(self.data['test_seen']['labels']).cpu().numpy()
        print("Sample sizes per class in test_seen:")
        for class_idx, count in enumerate(test_seen_class_counts):
            if count > 0:  # Only print classes with samples
                print(f"Class {class_idx}: {count} samples")

        test_unseen_class_counts = torch.bincount(self.data['test_unseen']['labels']).cpu().numpy()
        print("\nSample sizes per class in test_unseen:")
        for class_idx, count in enumerate(test_unseen_class_counts):
            if count > 0:  # Only print classes with samples
                print(f"Class {class_idx}: {count} samples")
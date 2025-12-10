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
        image_file = self.image_files[idx][0]
        if self.dataset == 'CUB':
            split_idx = 6
        elif self.dataset == 'SUN':
            split_idx = 7
        elif self.dataset == 'AWA2':
            split_idx = 5
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[split_idx:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def extract_features(config):

    img_dir = f'data/{config.dataset}'
    file_paths = f'data/xlsa17/data/{config.dataset}/res101.mat'        #存有image_files文件的路径，也就是每张图的路径。还有一个features，里面存放的是矩阵，还有labels标签
    save_path = f'data/{config.dataset}/feature_map_ResNet_101_{config.dataset}.hdf5'
    attribute_path = f'w2v/{config.dataset}_attribute.pkl'                               #是一个词向量的矩阵 形状是312，300

    # region feature extractor
    resnet101 = models.resnet101(pretrained=True).to(config.device)
    resnet101 = nn.Sequential(*list(resnet101.children())[:-2]).eval()

    data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    Dataset = CustomedDataset(config.dataset, img_dir, file_paths, data_transforms)
    dataset_loader = torch.utils.data.DataLoader(Dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)

    with torch.no_grad():
        all_features = []
        for _, imgs in enumerate(dataset_loader):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            all_features.append(features.cpu().numpy())
        all_features = np.concatenate(all_features, axis=0)

    # get remaining metadata
    matcontent = Dataset.matcontent
    labels = matcontent['labels'].astype(int).squeeze() - 1

    split_path = os.path.join(f'data/xlsa17/data/{config.dataset}/att_splits.mat')      #有所有类别的名字 也就是文件夹名，还有一个矩阵，名叫att
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
    if config.dataset == 'CUB':
        assert w2v_att.shape == (312,300)      #  312个属性，200个类别
    elif config.dataset == 'SUN':
        assert w2v_att.shape == (102,300)
    elif config.dataset == 'AWA2':
        assert w2v_att.shape == (85,300)

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', '-d', type=str, default='AWA2')
    # parser.add_argument('--compression', '-c', action='store_true', default=False)
    # parser.add_argument('--batch_size', '-b', type=int, default=200)
    # parser.add_argument('--device', '-g', type=str, default='cuda:0')
    # parser.add_argument('--nun_workers', '-n', type=int, default='16')
    # config = parser.parse_args()
    # extract_features(config)

    # img_dir = f'data/{config.dataset}'
    # file_paths = f'data/xlsa17/data/{config.dataset}/res101.mat'
    # save_path = f'data/CUB/feature_map_ResNet_101_CUB.hdf5'

    # attribute_path = f'w2v/wurenji_attribute.pkl'
    # # construct attribute w2v
    # with open(attribute_path,'rb') as f:
    #     w2v_att = pickle.load(f)
    #     # print(w2v_att)
    #     print(w2v_att.shape)
    #     assert w2v_att.shape == (200, 300)

    att_path= os.path.join(f'data/xlsa17/data/wurenji/att_wurenji_splits.mat')
    att_path2 = os.path.join(f'data/xlsa17/data/CUB/att_splits.mat')


    res101_wurenji = os.path.join(f'data/xlsa17/data/wurenji/res101.mat')
    res101_cub = os.path.join(f'data/xlsa17/data/CUB/res101.mat')

    matcontent_wurenji = sio.loadmat(att_path)
    matconten2 = sio.loadmat(res101_wurenji )
    # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    # 从matconten2中获取test_unseen_loc，test_seen_loc和trainval_loc的值


    # test_unseen_loc = matconten2['test_unseen_loc']
    # test_seen_loc = matconten2['test_seen_loc']
    # trainval_loc = matconten2['trainval_loc']


    # matcontent_wurenji['test_unseen_loc'] = test_unseen_loc
    # matcontent_wurenji['test_seen_loc'] = test_seen_loc
    # matcontent_wurenji['trainval_loc'] = trainval_loc
    #
    # sio.savemat('att_wurenji_splits.mat', matcontent_wurenji)
    # np.set_printoptions(threshold=np.inf)
    print(matcontent_wurenji)
    # print(matconten2)


    # print( matcontent_wurenji["original_att"][:, 1])
    # print(matcontent_wurenji["original_att"][:, 0].shape)
    # print(matcontent_wurenji["att"].shape)
    # print(matcontent_wurenji["test_unseen_loc"])

    # test_seen_loc_array = np.array(matcontent_wurenji["test_seen_loc"])
    #
    # # 找到最小的数字值
    # min_value = np.min(test_seen_loc_array)    #min  2502,701
    # #
    # print("最小的数字值是:", min_value)

    # print( matcontent_wurenji['image_files'])
    # print(matconten2["test_unseen_loc"])
    # print(matconten2["test_seen_loc"])
    # print(matconten2["trainval_loc"].shape)

    # print(matconten2["att"])
    # print(matconten_cub)

    # res101::::matcontent["labels"].shape ===(11788, 1)  matcontent["features"].shape===(2048, 11788)  matcontent['image_files'].shape===(11788, 1)

    # att_splits:::::allclasses_names===(200, 1)   att==(312, 200)    original_att==(312, 200)   test_unseen_loc==(2967, 1)    test_seen_loc==(1764, 1)  trainval_loc== (7057, 1)
    # with h5py.File(save_path, 'r') as f:
    #     # 打印文件中所有的组
    #     print("Groups in HDF5 file:")
    #     for group in f.keys():
    #         print(group)
    #
    #     # 打印文件中所有的数据集
    #     print("\nDatasets in HDF5 file:")
    #     for dataset in f:
    #         print(dataset)

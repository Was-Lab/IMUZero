import torch
import torch.optim as optim
import torch.nn as nn
from model import TransZero
from dataset import imuDataLoader
from helper_func import eval_zs_gzsl
import numpy as np
import torch.nn.functional as F


def random_permute_axes(data):
    """
    对数据的每个 (x, y, z) 组进行随机转换。
    参数:
    返回:
        data (torch.Tensor): 形状为 (batch_size, 170, 27)
        permuted_data (torch.Tensor): 转换后的数据
        perm (list): 变换的顺序
    """

    assert data.shape[2] % 9 == 0, "数据的最后一个维度必须是9的倍数。"

    # 随机生成一个 (x, y, z) 的排列，但不能是原来的顺序
    axes = [0, 1, 2]
    perm = axes.copy()
    while perm == axes:
        np.random.shuffle(perm)

    # 对数据进行转换
    # print('perm', perm)
    batch_size, time_steps, features = data.shape
    # print('batch_size', batch_size)
    # print('time_steps', time_steps)
    data = data.view(batch_size, time_steps, -1, 3)  # (batch_size, 170, 9, 3)
    permuted_data = data[:, :, :, perm]  # 应用随机转换
    permuted_data = permuted_data.view(batch_size, time_steps, -1)  # 恢复原始形状

    return permuted_data, perm


def random_permute_axes_mhealth(data):
    """
    对 mHealth 数据中的传感器 (x, y, z) 组进行随机转换。
    参数:
        data (torch.Tensor): 形状为 (batch_size, 60, 21)
    返回:
        permuted_data (torch.Tensor): 转换后的数据
        perms (list): 每个传感器的变换顺序
    """

    assert data.shape[2] == 21, "数据的最后一个维度必须是21。"

    # 每个传感器的通道数量
    sensor_channels = [3, 9, 9]

    # 为每个传感器生成随机排列
    perms = []
    for channels in sensor_channels:
        axes = list(range(channels))
        perm = axes.copy()
        while perm == axes:
            np.random.shuffle(perm)
        perms.append(perm)

    # 对数据进行转换
    batch_size, time_steps, features = data.shape

    # 根据传感器通道数分割数据并应用转换
    start_idx = 0
    permuted_data = []
    for channels, perm in zip(sensor_channels, perms):
        sensor_data = data[:, :, start_idx:start_idx + channels]
        sensor_data = sensor_data[:, :, perm]
        permuted_data.append(sensor_data)
        start_idx += channels

    # 将所有转换后的数据合并回原始形状
    permuted_data = torch.cat(permuted_data, dim=2)

    return permuted_data, perms


def compute_similarity_loss(original_features, permuted_features):
    """
    计算原始数据和转换后数据的特征相似性损失。
    参数:
        original_features (torch.Tensor): 原始数据的特征
        permuted_features (torch.Tensor): 转换后数据的特征




    返回:
        loss (torch.Tensor): 特征相似性损失
    """
    return F.mse_loss(original_features, permuted_features)
# 修改gzsl，不要value。
# 放入model函数中，要先用Config

# Load configuration from a YAML file
# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
#
# config_path = 'wandb_config/wurenji.yaml'
# with open(config_path, 'r') as file:
#     config_dict = yaml.safe_load(file)
#
# # config = Config(**config_dict)
# config = config_dict



# load dataset
# dataloader = wurenjiDataLoader('.', config['device'], is_balance=False)

dataloader = imuDataLoader('cuda:2', is_balance=False)

# set random seed
# seed = config['random_seed']
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# TransZero model
# model = TransZero(Config(**config_dict), dataloader.att, dataloader.w2v_att,
#                   dataloader.seenclasses, dataloader.unseenclasses).to(config['device'])
model = TransZero(dataloader.att, dataloader.w2v_att,
                  dataloader.seenclasses, dataloader.unseenclasses).to('cuda:2')
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)

max_acc_train_seen = float('-inf')
max_best_performance_zsl = float('-inf')
max_best_performance = [float('-inf'), float('-inf'), float('-inf')]

# main loop
# niters = dataloader.ntrain * config['epochs']//config['batch_size']
# report_interval = niters//config['epochs']
niters = dataloader.ntrain * 800//22
report_interval = niters//800
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    # batch_label, batch_feature, batch_att = dataloader.next_batch(config['batch_size'])
    batch_label, batch_imu, batch_att = dataloader.next_batch(22)
    # # batch_feature = batch_feature.unsqueeze(3)
    # out_package = model(batch_imu)
    #
    # in_package = out_package
    # in_package['batch_label'] = batch_label
    #
    # out_package = model.compute_loss(in_package)
    # loss, loss_CE, loss_cal, loss_reg = out_package['loss'], out_package['loss_CE'], out_package['loss_cal'], \
    # out_package['loss_reg']
    #
    # loss.backward()
    # optimizer.step()

    # 对 (x, y, z) 进行随机转换
    permuted_batch_imu, perm = random_permute_axes(batch_imu)
    # permuted_batch_imu, perm = random_permute_axes_mhealth(batch_imu)

    # 输入模型
    original_out_package = model(batch_imu)
    permuted_out_package = model(permuted_batch_imu)

    # 计算特征相似性损失
    similarity_loss = compute_similarity_loss(original_out_package['embed'], permuted_out_package['embed'])

    # 计算其他损失并添加相似性损失
    in_package = original_out_package
    in_package['batch_label'] = batch_label
    out_package = model.compute_loss(in_package)
    loss, loss_CE, loss_cal, loss_reg = out_package['loss'], out_package['loss_CE'], out_package['loss_cal'], \
    out_package['loss_reg']

    # 总损失
    total_loss = loss + similarity_loss  # similarity_loss_weight 是相似性损失的权重

    # 反向传播和参数更新
    total_loss.backward()
    # loss.backward()
    optimizer.step()

    # if i % report_interval == 0:
    #
    #     # acc_seen, acc_novel, H, acc_zs, acc_train_seen = eval_zs_gzsl(dataloader, model, config['device'],
    #     #                                                               batch_size=config['batch_size'])
    #     acc_seen, acc_novel, H, acc_zs, acc_train_seen = eval_zs_gzsl(dataloader, model, 'cuda:1',
    #                                                                   batch_size=22)
    #
    #     # if H > best_performance[2]:
    #     best_performance = [acc_novel, acc_seen, H, acc_zs]
    #     # if acc_zs > best_performance_zsl:
    #     best_performance_zsl = acc_zs
    #
    #     if acc_train_seen > max_acc_train_seen:
    #         max_acc_train_seen = acc_train_seen
    #     if best_performance_zsl > max_best_performance_zsl:
    #         max_best_performance_zsl = best_performance_zsl
    #     for j in range(3):
    #         if best_performance[j] > max_best_performance[j]:
    #             max_best_performance[j] = best_performance[j]
    #
    #     print('-' * 30)
    #     print(
    #         'iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, loss_reg=%.3f | acc_train_seen=%.3f | acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f' % (
    #             i, int(i // report_interval),
    #             loss.item(), loss_CE.item(), loss_cal.item(),
    #             loss_reg.item(),
    #             max_acc_train_seen,
    #             max_best_performance[0], max_best_performance[1],
    #             max_best_performance[2], max_best_performance_zsl))


    # report result
    if i % report_interval == 0:
        print('-'*30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            dataloader, model, 'cuda:2', batch_size=22)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
              'loss_reg=%.3f | acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | '
              'acc_zs=%.3f' % (
                  i, int(i//report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(),
                  loss_reg.item(),
                   best_performance[0], best_performance[1],
                  best_performance[2], best_performance_zsl))

        # wandb.log({
        #     'iter': i,
        #     'loss': loss.item(),
        #     'loss_CE': loss_CE.item(),
        #     'loss_cal': loss_cal.item(),
        #     'loss_reg': loss_reg.item(),
        #     'acc_unseen': acc_novel,
        #     'acc_seen': acc_seen,
        #     'H': H,
        #     'acc_zs': acc_zs,
        #     'best_acc_unseen': best_performance[0],
        #     'best_acc_seen': best_performance[1],
        #     'best_H': best_performance[2],
        #     'best_acc_zs': best_performance_zsl
        # })


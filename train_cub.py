import torch
import torch.optim as optim
import torch.nn as nn
from model import TransZero
from dataset import CUBDataLoader
from helper_func import eval_zs_gzsl
import numpy as np
import yaml  # Import the yaml package

# 修改gzsl，不要value。
# 放入model函数中，要先用Config

# Load configuration from a YAML file
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config_path = 'wandb_config/cub_gzsl.yaml'
with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)

# config = Config(**config_dict)
config = config_dict



# load dataset
dataloader = CUBDataLoader('.', config['device'], is_balance=False)

# set random seed
seed = config['random_seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# TransZero model
model = TransZero(Config(**config_dict), dataloader.att, dataloader.w2v_att,
                  dataloader.seenclasses, dataloader.unseenclasses).to(config['device'])
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# main loop
niters = dataloader.ntrain * config['epochs']//config['batch_size']
report_interval = niters//config['epochs']
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    batch_label, batch_feature, batch_att = dataloader.next_batch(
        config['batch_size'])
    out_package = model(batch_feature)

    in_package = out_package
    in_package['batch_label'] = batch_label

    out_package = model.compute_loss(in_package)
    loss, loss_CE, loss_cal, loss_reg = out_package['loss'], out_package[
        'loss_CE'], out_package['loss_cal'], out_package['loss_reg']

    loss.backward()
    optimizer.step()

    # report result
    if i % report_interval == 0:
        print('-'*30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            dataloader, model, config['device'], batch_size=config['batch_size'])


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
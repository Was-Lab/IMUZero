import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms,models
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import scipy.io as sio
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 1. 加载语义特征
semantic_path = 'row_3d2_attribute.mat'
semantic_features = sio.loadmat(semantic_path)['features']  # 调整键名以匹配实际情况

semantic_features = torch.from_numpy(semantic_features).float()
semantic_features = semantic_features.reshape(25, -1)  # 将特征重新塑形为[25, 4608]
print(semantic_features.shape)
# sim_cos = F.cosine_similarity(semantic_features.unsqueeze(1),semantic_features.unsqueeze(0),dim=2)
sim_cos = F.cosine_similarity(semantic_features[0],semantic_features[3],dim=0)
print(sim_cos)
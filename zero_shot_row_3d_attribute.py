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
import math
from resnet import ResNet18
# 1. 加载语义特征
semantic_path = 'row_3d2_attribute.mat'
semantic_features = sio.loadmat(semantic_path)['features']  # 调整键名以匹配实际情况
semantic_features = torch.from_numpy(semantic_features).float()
semantic_features = semantic_features.reshape(25, -1)  # 将特征重新塑形为[25, 4608]
# semantic_features=semantic_features[:5,]
class_order = ["T1000", "T1110", "T10110", "T0000", "T1101", "T1001", "T1010", "T0011", "T0110", "T0111",
               "T0100", "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111",
               "T10001", "T1100", "T1011", "T0010", "T10111"]




os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 数据加载与预处理
def load_images_from_folder(folder, count):
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Use filename here
            filenames.append(filename)
        if len(filenames) >= count:
            break
    return filenames


def load_images_from_folder(folder, count):
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Use filename here
            filenames.append(filename)
        if len(filenames) >= count:
            break
    return filenames


def create_dataset(root_dir, transform, train_count=450, val_count=150):
    # 指定类别的顺序
    # class_order = ["T1000", "T1110", "T10110", "T0000", "T1101"]
    class_order = ["T1000", "T1110", "T10110", "T0000", "T1101", "T1001", "T1010", "T0011", "T0110", "T0111", "T0100",
                   "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111", "T10001",
                   "T1100", "T1011", "T0010", "T10111"]
    train_data = []
    val_data = []

    for cls in class_order:
        class_dir = os.path.join(root_dir, cls)
        if os.path.exists(class_dir):
            all_images = [f for f in os.listdir(class_dir) if f.endswith(".jpg") or f.endswith(".png")]

            # 打乱全部图片
            np.random.shuffle(all_images)

            train_images = all_images[:train_count]
            val_images = all_images[train_count:train_count + val_count]

            for img in train_images:
                train_data.append((os.path.join(class_dir, img), cls))

            for img in val_images:
                val_data.append((os.path.join(class_dir, img), cls))
        else:
            print(f"Directory not found: {class_dir}")

    train_dataset = [(x[0], class_order.index(x[1])) for x in train_data]
    val_dataset = [(x[0], class_order.index(x[1])) for x in val_data]

    class CustomDataset(datasets.VisionDataset):
        def __init__(self, data, transform=None):
            super().__init__(root=root_dir, transform=transform)
            self.data = data

        def __getitem__(self, index):
            img_path, target = self.data[index]
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, target

        def __len__(self):
            return len(self.data)

    train_dataset = CustomDataset(data=train_dataset, transform=transform)
    val_dataset = CustomDataset(data=val_dataset, transform=transform)

    return train_dataset, val_dataset

transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

root_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_600'
train_dataset, val_dataset = create_dataset(root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class model1(nn.Module):
    def __init__(self, semti):
        super(model1, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # 从指定路径加载预训练权重

        self.model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
        num_ftrs = self.model.fc.out_features
        self.semti = semti
        self.fc = nn.Linear(num_ftrs, 512)  # 将最后一层输出改为生成与语义特征同样数量的特征
        self.fc2 = nn.Linear( self.semti.size(1) , 512)
    def forward(self,x):
        x = self.model(x)
        x = self.fc(x)
        sem = self.fc2( self.semti)
        logits = x.mm(F.normalize(sem.t(),dim=1))
        return  logits

# 3. 定义模型、优化器和损失函数
model = model1(semantic_features.to(device)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 100
start_epoch = 0  # Add this line to keep track of the starting epoch
flag = "train"
best_accuracy = 0  # 初始化最好的准确性



if flag == "train":

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Train Accuracy123: {train_accuracy:.2f}%")

        if (epoch + 1) % 5 == 0:
            model.eval()  # Set model to evaluate mode
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():  # In evaluation phase, we don't compute gradients
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    # print(labels)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            accuracy = 100 * correct / total
            print(f'Validation accuracy after epoch {epoch+1}: {accuracy:.2f}%')





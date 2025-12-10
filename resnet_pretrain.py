'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
# import os
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch.autograd import Variable
# from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class PreActBlock(nn.Module):
#     '''Pre-activation version of the BasicBlock.'''
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += shortcut
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class PreActBottleneck(nn.Module):
#     '''Pre-activation version of the original Bottleneck module.'''
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out += shortcut
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=5):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = conv3x3(2, 64)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(512*block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x, return_feature=False):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # out = F.avg_pool2d(out, 4)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         if return_feature:
#             return out, y
#         else:
#             return y
#
#     # function to extact the multiple features
#     def feature_list(self, x):
#         out_list = []
#         out = F.relu(self.bn1(self.conv1(x)))
#         out_list.append(out)
#         out = self.layer1(out)
#         out_list.append(out)
#         out = self.layer2(out)
#         out_list.append(out)
#         out = self.layer3(out)
#         out_list.append(out)
#         out = self.layer4(out)
#         out_list.append(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y, out_list
#
#     # function to extact a specific feature
#     def intermediate_forward(self, x, layer_index):
#         out = F.relu(self.bn1(self.conv1(x)))
#         if layer_index == 1:
#             out = self.layer1(out)
#         elif layer_index == 2:
#             out = self.layer1(out)
#             out = self.layer2(out)
#         elif layer_index == 3:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#         elif layer_index == 4:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#             out = self.layer4(out)
#         return out
#
#     # function to extact the penultimate features
#     def penultimate_forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         penultimate = self.layer4(out)
#         out = F.avg_pool2d(penultimate, 4)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y, penultimate
#
#
# def ResNet18(num_c):
#     return ResNet(PreActBlock, [2,2,2,2], num_classes=num_c)
#
#
# def ResNet34(num_c):
#     return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)
#
#
# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])
#
#
# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])
#
#
# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])
#
#
# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

# test()

# 导入所需的库
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import product
from PIL import Image


# # 替换成你的路径
# data_dir = 'data/wurenji/classify'
#
# # 列出所有文件
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         print(os.path.join(root, file))



# 描述ResNet里的部分
# 请在这里插入之前给出的ResNet类定义，例如：
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)#图片经过transformer后，会经历升维
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        if return_feature:
            return out, y
        else:
            return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

# 例子：定义ResNet18模型
def ResNet18(num_c):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_c)
def ResNet34(num_c):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)


def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])


def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])
# 定义设备
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # 调整图像大小以适配模型
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 适用于RGB三个通道
])


def load_images_from_folder(folder, count):
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Use filename here
            filenames.append(filename)
        if len(filenames) >= count:
            break
    return filenames


def create_dataset(root_dir, transform, train_count=1000, val_count=250):
    classes = os.listdir(root_dir)
    train_data = []
    val_data = []

    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        all_images = [f for f in os.listdir(class_dir) if f.endswith(".jpg") or f.endswith(".png")]

        # 打乱全部图片
        np.random.shuffle(all_images)

        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]

        for img in train_images:
            train_data.append((os.path.join(class_dir, img), cls))

        for img in val_images:
            val_data.append((os.path.join(class_dir, img), cls))

    train_dataset = [(x[0], classes.index(x[1])) for x in train_data]
    val_dataset = [(x[0], classes.index(x[1])) for x in val_data]

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


root_dir = 'data/wurenji/classify_millon'
train_dataset, val_dataset = create_dataset(root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# print(val_loader)
# for images, labels in val_loader:
#     print(labels)
# 获取类别名称列表
class_names = os.listdir(root_dir)
# print(classes)

# 设置CUDA设备
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet101模型，这里我们指定pretrained=False因为我们手动加载权重
model = models.resnet101(pretrained=False)
# 从指定路径加载预训练权重
model.load_state_dict(torch.load('resnet101-63fe2227.pth'))

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 25)  # 适配到你的类别数目
model = model.to(device)

# 实例化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    此函数打印并绘制混淆矩阵。
    可以通过设置`normalize=True`来应用归一化。
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_checkpoint(epoch, model, optimizer, filename='data/wurenji/pth/pic/checkpoint_18npy_one.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)

num_epochs = 100
start_epoch = 0  # Add this line to keep track of the starting epoch

# Check if there exists a checkpoint file
if os.path.exists('data/wurenji/pth/pic/checkpoint_18npy_one.pth'):
    checkpoint = torch.load('data/wurenji/pth/pic/checkpoint_18npy_one.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

best_accuracy = 0  # 初始化最好的准确性

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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.2f}%")

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

        # 计算并绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cm, class_names, title='Confusion Matrix')
        plt.savefig(f'confusion_matrix_resnet_npy_{epoch}.png')  # 将图像保存为文件
        plt.show()

        # 保存模型权重-----------------------------------------------------------------------------------------------------
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved at epoch {epoch+1} with accuracy: {accuracy:.2f}%')
    # Save checkpoint after each epoch
    save_checkpoint(epoch, model, optimizer)
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product
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
print(semantic_features.shape)

class_order = ["T1000", "T1110", "T10110", "T0000", "T1101", "T1001", "T1010", "T0011", "T0110", "T0111", "T0100",
               "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111", "T10001",
               "T1100", "T1011", "T0010", "T10111"]


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 2. 数据加载与预处理
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小以适配模型
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 适用于RGB三个通道
])


def remap_classes(target, class_to_idx):
    """根据提供的类别序列重新映射类别标签"""
    # 通过查询字典 `class_to_idx` 来重新映射类别标签
    return class_to_idx[target]


class_to_idx = {cls_name: i for i, cls_name in enumerate(class_order)}

train_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_450_train'
val_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_150_test'
# 使用ImageFolder加载数据集
train_dataset = datasets.ImageFolder(train_dir, transform=transform,
                                     target_transform=lambda target: remap_classes(class_order[target], class_to_idx))
val_dataset = datasets.ImageFolder(val_dir, transform=transform,
                                   target_transform=lambda target: remap_classes(class_order[target], class_to_idx))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class model1(nn.Module):
    def __init__(self, semti):
        super(model1, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # 从指定路径加载预训练权重

        self.model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
        num_ftrs = self.model.fc.out_features
        self.semti = semti
        self.fc = nn.Linear(num_ftrs, 512)  # 将最后一层输出改为生成与语义特征同样数量的特征
        self.fc2 = nn.Linear(self.semti.size(1), 512)
        # __________________________________________________________________________________
        # self.n_classes=25
        # orth_vec = self.generate_random_orthogonal_matrix(512, 25)
        # i_nc_nc = torch.eye(self.n_classes)
        # one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.n_classes, self.n_classes), (1 / self.n_classes))
        # self.etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
        #                          math.sqrt(self.n_classes / (self.n_classes - 1))).cuda()

    # def generate_random_orthogonal_matrix(self, feat_in, num_classes):
    #     rand_mat = np.random.random(size=(feat_in, num_classes))
    #     orth_vec, _ = np.linalg.qr(rand_mat)
    #     orth_vec = torch.tensor(orth_vec).float()
    #     assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
    #         "The max irregular value is : {}".format(
    #             torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    #     return orth_vec

    # __________________________________________________________________________________
    def forward(self,x):
        x = self.model(x)
        x = self.fc(x)
        sem = self.fc2(self.semti)
        logits = x.mm(F.normalize(sem.t(), dim=1))
        return logits

root_dir = 'data/wurenji/wurenji_stft_pic_abs'
# 获取类别名称列表
class_names = os.listdir(root_dir)
# 示例用法:
# 创建一个类实例，指定分类数量
model = model1(semantic_features.to(device)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




# 4. 训练模型
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


def evaluate_accuracy(data_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            print(f"Labels type: {type(labels)}")  # 调试打印语句
            if isinstance(labels, tuple):  # 额外检查确保labels是期待的类型
                print("Warning: Labels are a tuple, expected a Tensor.")
                continue  # 或者你可以在这里做更复杂的处理，以应对这种情况

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the validation/test images: {accuracy * 100:.2f}%')

def save_checkpoint(epoch, model, optimizer, filename='data/wurenji/pth/zero_shot_3d/checkpoint_18npy_one.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)


# Check if there exists a checkpoint file
if os.path.exists('data/wurenji/pth/zero_shot_3d/checkpoint_18npy_one.pth'):
    checkpoint = torch.load('data/wurenji/pth/zero_shot_3d/checkpoint_18npy_one.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")


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
            plt.savefig(f'data/wurenji/pth/zero_shot_3d/confusion_matrix_resnet_npy_{epoch}.png')  # 将图像保存为文件
            plt.show()

            # 保存模型权重-----------------------------------------------------------------------------------------------------
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'data/wurenji/pth/zero_shot_3d/best_model.pth')
                print(f'Model saved at epoch {epoch+1} with accuracy: {accuracy:.2f}%')
        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer)

# /////////////////////////////////////////////////////////////////////////////////////////直接做测试集
else:
    # 加载模型
    model = models.resnet18(pretrained=False)

    # 修改全连接层以匹配你的类别数（假设为25）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 25)

    # 加载预训练权重，这里先假设权重已经存放在指定路径
    model_path = 'data/wurenji/pth/zero_shot_3d/best_model.pth'
    state_dict = torch.load(model_path)

    # 创建一个新的state_dict，其键值不包含'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 排除‘module.’
        new_state_dict[name] = v

    # 加载调整后的state_dict到模型
    model.load_state_dict(new_state_dict)

    evaluate_accuracy(val_loader, model)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
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
import pandas as pd
import math
from resnet import ResNet18


# 1. 加载语义特征
semantic_path = 'row_3d2_attribute.mat'
semantic_features = sio.loadmat(semantic_path)['features']  # 调整键名以匹配实际情况
semantic_features = torch.from_numpy(semantic_features).float()
semantic_features = semantic_features.reshape(25, -1)  # 将特征重新塑形为[25, 4608]
# semantic_features=semantic_features[:5,]
# for row in semantic_features:
#     print(row)

class_order = ["T1000", "T1110", "T10110", "T0000", "T1101", "T1001", "T1010", "T0011", "T0110", "T0111", "T0100",
               "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111", "T10001",
               "T1100", "T1011", "T0010", "T10111"]
# , "T1001", "T1010", "T0011", "T0110", "T0111", "T0100",
#                "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111"
class_order_unseen = ["T10001","T1100", "T1011", "T0010", "T10111"]

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# 2. 数据加载与预处理
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小以适配模型
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 适用于RGB三个通道
])

train_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_450_train'
val_dir = 'data/wurenji/wurenji_stft_pic_abs_cut_150_test'



class CustomDataset(Dataset):
    dataset_classes = []
    current_classes = []

    def __init__(self, root_dir, type, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Loading classes according to class_order sequence
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        class_order = ["T1000", "T1110", "T10110", "T0000", "T1101", "T1001", "T1010", "T0011", "T0110", "T0111", "T0100",
                       "T10000", "T11000", "T10011", "T0001", "T0101", "T10100", "T10101", "T10010", "T1111", "T10001",
                       "T1100", "T1011", "T0010", "T10111"]

        if type == "seen":
            self.classes = [cls for cls in class_order if cls in all_classes][:20]
        else:
            self.classes = [cls for cls in class_order if cls in all_classes][20:]

        CustomDataset.dataset_classes = all_classes
        CustomDataset.current_classes = self.classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.img_paths = self._get_img_paths()

    def _get_img_paths(self):
        img_paths = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            img_paths.extend([(os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]) for img_name in os.listdir(cls_dir)])
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, target = self.img_paths[idx]
        # print(img_path,target)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Return target as tensor
        target = torch.tensor(target)
        return img, target

train_dataset = CustomDataset(train_dir,type="seen", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_dir,type="seen", transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print(train_dataset.current_classes)
# # 获取类别名称列表
class_names=train_dataset.current_classes
print(class_names)
unseen_val_dataset = CustomDataset(val_dir,"unseen",transform=transform)
unseen_val_loader = DataLoader(unseen_val_dataset,batch_size=110,shuffle=False)

# for images, targets in unseen_val_loader:
#     # images 是图像数据，targets 是标签数据
#     print("图像数据:", images)
#     print("标签数据:", targets)


# class_order_reordered = [class_ for class_ in train_dataset.dataset_classes]
# semantic_features = [semantic_features[train_dataset.dataset_classes.index(class_)] for class_ in class_order_reordered]
# # 使用 torch.cat() 将张量按行拼接
# merged_tensor = torch.cat(semantic_features, dim=0)
# # 调整形状成 (25, n)
# semantic_features = merged_tensor.view(25, -1)
#
# # 打印结果
# print("Reordered class_order:", class_order_reordered)
# print("Reordered semantic_features:")
# for row in semantic_features:
#     print(row)
#

# root_dir = 'data/wurenji/wurenji_stft_pic_abs'
# class_names = os.listdir(root_dir)
# class_names = class_order_seen


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
        sem = self.fc2( self.semti)
        logits = x.mm(F.normalize(sem.t(),dim=1))
        return  logits

    def extract_features(self, x):
        # Extract features without final linear layer
        x = self.model(x)
        x = self.fc(x)
        sem = self.fc2(self.semti[20:,])
        logits = x.mm(F.normalize(sem.t(), dim=1))
        return logits

class model2(nn.Module):
    def __init__(self, num_classes=1000):  # Default number of classes in ImageNet
        super(model2, self).__init__()
        # Load a pretrained ResNet-18 model
        self.model = models.resnet18(pretrained=False)

        # Load pretrained weights
        pretrained_dict = torch.load('resnet18-f37072fd.pth', map_location='cpu')  # Ensure the right device is used
        self.model.load_state_dict(pretrained_dict)

        # Change the final fully connected layer to match the number of required classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the network
        return self.model(x)

# 示例用法:
# 创建一个类实例，指定分类数量
model = model1(semantic_features.to(device)).to(device)
# model = model2(num_classes=5).to(device)




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
            images, labels = images.to(device), labels.to(device)
            print("seen")
            print(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the  seen validation images: {accuracy * 100:.2f}%')

def save_checkpoint(epoch, model, optimizer, filename='data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/checkpoint_18npy_one.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)


num_epochs = 100
start_epoch = 0  # Add this line to keep track of the starting epoch
flag = "trai"
best_accuracy = 0  # 初始化最好的准确性
# Check if there exists a checkpoint file
if os.path.exists('data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/checkpoint_18npy_one.pth'):
    checkpoint = torch.load('data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/checkpoint_18npy_one.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")





def evaluate_accuracy_unseen(data_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            print("unseen")
            print(labels)
            # print(f"Labels type: {type(labels)}")  # 调试打印语句
            # if isinstance(labels, tuple):  # 额外检查确保labels是期待的类型
            #     print("Warning: Labels are a tuple, expected a Tensor.")
            #     continue  # 或者你可以在这里做更复杂的处理，以应对这种情况

            outputs = model.extract_features(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            print(predicted)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the  unseen validation  images: {accuracy * 100:.2f}%')

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
            plt.savefig(f'data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/confusion_matrix_resnet_npy_{epoch}.png')  # 将图像保存为文件
            plt.show()

            # 保存模型权重-----------------------------------------------------------------------------------------------------
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/best_model.pth')
                print(f'Model saved at epoch {epoch+1} with accuracy: {accuracy:.2f}%')
        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer)

# /////////////////////////////////////////////////////////////////////////////////////////直接做测试集
else:
    # 加载预训练权重，这里先假设权重已经存放在指定路径
    model_path = 'data/wurenji/pth/zero_shot_unseen_20_order_by_dataset/best_model.pth'
    state_dict = torch.load(model_path)

    # 创建一个新的state_dict，其键值不包含'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 排除‘module.’
        new_state_dict[name] = v

    # 加载调整后的state_dict到模型
    model.load_state_dict(new_state_dict)
    evaluate_accuracy(val_loader , model)
    # evaluate_accuracy_unseen(unseen_val_loader , model)
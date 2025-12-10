import torch
import torch.nn.functional as F
from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(test_label, predicted_label, classes):
    # 计算混淆矩阵
    cm = confusion_matrix(test_label.cpu().numpy(), predicted_label.cpu().numpy(), labels=classes.cpu().numpy())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes.cpu().numpy(),
                yticklabels=classes.cpu().numpy())

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Unseen Classes')
    plt.show()

def val_gzsl(test_X, test_label, target_classes,in_package,bias = 0):

    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)

            out_package = model(input)

            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
        print(f"Label {classes[i].item()} is mapped to index {i}")

    return mapped_label

# def val_zs_gzsl(test_X, test_label, unseen_classes,in_package,bias = 0):
#     batch_size = in_package['batch_size']
#     model = in_package['model']
#     device = in_package['device']
#     with torch.no_grad():
#         start = 0
#         ntest = test_X.size()[0]
#         print('ntest',ntest)
#         predicted_label_gzsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl_t = torch.LongTensor(test_label.size())
#         for i in range(0, ntest, batch_size):
#
#             end = min(ntest, start+batch_size)
#
#             input = test_X[start:end].to(device)
#             # print('input',input.shape)
#
#             out_package = model(input)
#             output = out_package['S_pp']  # (22, 12)
#             # print('output',torch.argmax(output.data[:,unseen_classes], 1) )
#
#             output_t = output.clone()
#             output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
#             predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
#             predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1)
#
#             output[:,unseen_classes] = output[:,unseen_classes]+bias
#             predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
#
#
#             start = end
#
#
#         print(predicted_label_zsl)
#         print(predicted_label_zsl_t)
#         print(test_label)
#         acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
#         acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
#         acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
#
#         return acc_gzsl,acc_zs


# def val_zs_gzsl(test_X, test_label, unseen_classes, in_package, bias=0):
#     batch_size = in_package['batch_size']
#     model = in_package['model']
#     device = in_package['device']
#     with torch.no_grad():
#         start = 0
#         ntest = test_X.size()[0]
#         print('ntest', ntest)
#         predicted_label_gzsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl_t = torch.LongTensor(test_label.size())
#
#         for i in range(0, ntest, batch_size):
#             end = min(ntest, start + batch_size)
#             input = test_X[start:end].to(device)
#
#             out_package = model(input)
#             output = out_package['S_pp']
#
#             output_t = output.clone()
#             output_t[:, unseen_classes] += torch.max(output) + 1
#             predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
#             predicted_label_zsl_t[start:end] = torch.argmax(output.data[:, unseen_classes], 1)
#
#             output[:, unseen_classes] += bias
#             predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
#
#             start = end
#
#         acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
#         acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
#         acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t,
#                                          unseen_classes.size(0))
#
#         # 绘制混淆矩阵
#         plot_confusion_matrix(test_label, predicted_label_zsl, unseen_classes)
#
#         return acc_gzsl, acc_zs


def val_zs_gzsl(test_X, test_label, unseen_classes, in_package, bias=0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']

    all_features = []  # 用于存储特征
    all_labels = []  # 用于存储标签

    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        print('ntest', ntest)
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())

        for i in range(0, ntest, batch_size):
            end = min(ntest, start + batch_size)
            input = test_X[start:end].to(device)

            out_package = model(input)
            output = out_package['S_pp']  # (22, 12)

            # 获取嵌入特征
            features = out_package['embed']  # 假设有一个 'embed' 键
            all_features.append(features.cpu())
            all_labels.append(test_label[start:end].cpu())

            output_t = output.clone()
            output_t[:, unseen_classes] = output_t[:, unseen_classes] + torch.max(output) + 1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            output[:, unseen_classes] = output[:, unseen_classes] + bias
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)

            start = end

        # # 合并所有特征和标签
        # all_features = torch.cat(all_features)
        # all_labels = torch.cat(all_labels)
        #
        # # 进行 t-SNE 降维
        # tsne = TSNE(n_components=2, random_state=42)
        # features_2d = tsne.fit_transform(all_features.numpy())
        #
        # # 绘制 t-SNE 图
        # colors = cm.get_cmap('viridis', len(unseen_classes))
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels.numpy(), cmap=colors, alpha=0.6)
        # handles, _ = scatter.legend_elements()
        # # plt.legend(handles, [f'Class {cls.item()}' for cls in unseen_classes], title="Classes", loc="best")
        #
        # # 自动更新图片标题
        # plt.title(f't-SNE visualization of unseen classes2')
        # plt.axis('off')  # 去掉坐标轴和网格线
        # plt.show()

        acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
        acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t,
                                         unseen_classes.size(0))

        return acc_gzsl, acc_zs


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):  # gzsl

    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):

        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
    return per_class_accuracies.mean().item()



def eval_zs_gzsl(dataloader, model, device, bias_seen=0, bias_unseen=0, batch_size=50):
    model.eval()
    test_seen_imu = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)

    test_unseen_imu = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)

    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses

    # 打印 seen 类别和样本量统计
    seen_label_np = test_seen_label.cpu().numpy()
    print("\n[Seen Classes Sample Count]")
    for cls in seenclasses.cpu().numpy():
        count = np.sum(seen_label_np == cls)
        print(f"Class {cls}: {count} samples")

    # 打印 unseen 类别和样本量统计
    unseen_label_np = test_unseen_label.cpu().numpy()
    print("\n[Unseen Classes Sample Count]")
    for cls in unseenclasses.cpu().numpy():
        count = np.sum(unseen_label_np == cls)
        print(f"Class {cls}: {count} samples")

    in_package = {'model': model, 'device': device, 'batch_size': batch_size}

    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_imu, test_seen_label, seenclasses, in_package, bias=bias_seen)
        acc_novel, acc_zs = val_zs_gzsl(test_unseen_imu, test_unseen_label, unseenclasses, in_package, bias=bias_unseen)

    if (acc_seen + acc_novel) > 0:
        H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    else:
        H = 0

    return acc_seen, acc_novel, H, acc_zs

# def val_gzsl(test_X, test_label, target_classes, in_package, bias=0):
#     batch_size = in_package['batch_size']
#     model = in_package['model']
#     device = in_package['device']
#     with torch.no_grad():
#         start = 0
#         ntest = test_X.size()[0]
#         predicted_label = torch.LongTensor(test_label.size())
#         for i in range(0, ntest, batch_size):
#             end = min(ntest, start + batch_size)
#             input = test_X[start:end].to(device)
#             out_package = model(input)
#             output = out_package['S_pp']
#             output[:, target_classes] = output[:, target_classes] + bias
#             predicted_label[start:end] = torch.argmax(output.data, 1)
#             start = end
#
#         # 使用F1评估指标替代原来的准确率
#         f1_score = compute_f1_score_gzsl(test_label, predicted_label, target_classes, in_package)
#         return f1_score
#
#
# def map_label(label, classes):
#     mapped_label = torch.LongTensor(label.size()).fill_(-1)
#     for i in range(classes.size(0)):
#         mapped_label[label == classes[i]] = i
#         print(f"Label {classes[i].item()} is mapped to index {i}")
#     return mapped_label
#
#
# def val_zs_gzsl(test_X, test_label, unseen_classes, in_package, bias=0):
#     batch_size = in_package['batch_size']
#     model = in_package['model']
#     device = in_package['device']
#
#     all_features = []  # 用于存储特征
#     all_labels = []  # 用于存储标签
#
#     with torch.no_grad():
#         start = 0
#         ntest = test_X.size()[0]
#         print('ntest', ntest)
#         predicted_label_gzsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl = torch.LongTensor(test_label.size())
#         predicted_label_zsl_t = torch.LongTensor(test_label.size())
#
#         for i in range(0, ntest, batch_size):
#             end = min(ntest, start + batch_size)
#             input = test_X[start:end].to(device)
#
#             out_package = model(input)
#             output = out_package['S_pp']
#
#             # 获取嵌入特征
#             features = out_package['embed']  # 假设有一个 'embed' 键
#             all_features.append(features.cpu())
#             all_labels.append(test_label[start:end].cpu())
#
#             output_t = output.clone()
#             output_t[:, unseen_classes] = output_t[:, unseen_classes] + torch.max(output) + 1
#             predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
#             output[:, unseen_classes] = output[:, unseen_classes] + bias
#             predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
#             predicted_label_zsl_t[start:end] = torch.argmax(output.data[:, unseen_classes], 1)
#
#             start = end
#
#         # t-SNE 可视化代码保留
#         # 合并所有特征和标签
#         all_features = torch.cat(all_features)
#         all_labels = torch.cat(all_labels)
#
#         # 进行 t-SNE 降维
#         tsne = TSNE(n_components=2, random_state=42)
#         features_2d = tsne.fit_transform(all_features.numpy())
#
#         # 绘制 t-SNE 图
#         colors = cm.get_cmap('viridis', len(unseen_classes))
#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels.numpy(), cmap=colors, alpha=0.6)
#         handles, _ = scatter.legend_elements()
#         plt.title(f't-SNE visualization of unseen classes2')
#         plt.axis('off')
#
#         # 计算F1分数而非准确率
#         f1_gzsl = compute_f1_score_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
#         f1_zs = compute_f1_score_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
#         f1_zs_t = compute_f1_score(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
#
#         return f1_gzsl, f1_zs
#
#
# # 新增：计算每个类的F1分数
# def compute_f1_score(test_label, predicted_label, nclass):
#     f1_per_class = torch.FloatTensor(nclass).fill_(0)
#
#     for i in range(nclass):
#         # 找出属于该类别的样本索引
#         idx = (test_label == i)
#
#         # 真正例(TP)：预测为i且实际为i的样本数
#         tp = torch.sum((predicted_label[idx] == i).float())
#
#         # 假正例(FP)：预测为i但实际不是i的样本数
#         fp = torch.sum((predicted_label == i).float()) - tp
#
#         # 假负例(FN)：实际为i但预测不是i的样本数
#         fn = torch.sum(idx.float()) - tp
#
#         # 计算精确率和召回率
#         precision = tp / (tp + fp) if tp + fp > 0 else 0
#         recall = tp / (tp + fn) if tp + fn > 0 else 0
#
#         # 计算F1分数
#         f1_per_class[i] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
#
#     # 返回平均F1分数
#     return f1_per_class.mean().item()
#
#
# # 新增：计算GZSL设置下的F1分数
# def compute_f1_score_gzsl(test_label, predicted_label, target_classes, in_package):
#     device = in_package['device']
#     f1_per_class = torch.zeros(target_classes.size()[0]).float().to(device).detach()
#
#     predicted_label = predicted_label.to(device)
#
#     for i in range(target_classes.size()[0]):
#         target_class = target_classes[i]
#
#         # 真实标签为target_class的样本
#         is_class = (test_label == target_class)
#
#         # 预测标签为target_class的样本
#         pred_is_class = (predicted_label == target_class)
#
#         # 真正例(TP)：预测为target_class且实际为target_class的样本数
#         tp = torch.sum((predicted_label[is_class] == target_class).float())
#
#         # 假正例(FP)：预测为target_class但实际不是target_class的样本数
#         fp = torch.sum(pred_is_class.float()) - tp
#
#         # 假负例(FN)：实际为target_class但预测不是target_class的样本数
#         fn = torch.sum(is_class.float()) - tp
#
#         # 计算精确率和召回率
#         precision = tp / (tp + fp) if tp + fp > 0 else 0
#         recall = tp / (tp + fn) if tp + fn > 0 else 0
#
#         # 计算F1分数
#         f1_per_class[i] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
#
#     # 返回平均F1分数
#     return f1_per_class.mean().item()
#
#
# def eval_zs_gzsl(dataloader, model, device, bias_seen=0, bias_unseen=0, batch_size=50):
#     model.eval()
#     test_seen_imu = dataloader.data['test_seen']['resnet_features']
#     test_seen_label = dataloader.data['test_seen']['labels'].to(device)
#
#     test_unseen_imu = dataloader.data['test_unseen']['resnet_features']
#     test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
#
#     seenclasses = dataloader.seenclasses
#     unseenclasses = dataloader.unseenclasses
#
#     in_package = {'model': model, 'device': device, 'batch_size': batch_size}
#
#     with torch.no_grad():
#         f1_seen = val_gzsl(test_seen_imu, test_seen_label, seenclasses, in_package, bias=bias_seen)
#         f1_novel, f1_zs = val_zs_gzsl(test_unseen_imu, test_unseen_label, unseenclasses, in_package, bias=bias_unseen)
#
#     # 计算F1分数的调和平均值，替代原来的H值计算
#     if (f1_seen + f1_novel) > 0:
#         H = (2 * f1_seen * f1_novel) / (f1_seen + f1_novel)
#     else:
#         H = 0
#
#     return f1_seen, f1_novel, H, f1_zs

def val_gzsl_k(k,test_X, test_label, target_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        test_label = F.one_hot(test_label, num_classes=n_classes)
        predicted_label = torch.LongTensor(test_label.size()).fill_(0).to(test_label.device)
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            
            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label[start:end] = predicted_label[start:end].scatter_(1,idx_k,1)
            start = end
        
        acc = compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package)
        return acc


def val_zs_gzsl_k(k,test_X, test_label, unseen_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        
        test_label_gzsl = F.one_hot(test_label, num_classes=n_classes)
        predicted_label_gzsl = torch.LongTensor(test_label_gzsl.size()).fill_(0).to(test_label.device)
        
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            output = out_package['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label_gzsl[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label_gzsl[start:end] = predicted_label_gzsl[start:end].scatter_(1,idx_k,1)
            
            start = end
        
        acc_gzsl = compute_per_class_acc_gzsl_k(test_label_gzsl, predicted_label_gzsl, unseen_classes, in_package)
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,-1


def compute_per_class_acc_k(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    

def compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)
    
    hit = test_label*predicted_label
    for i in range(target_classes.size()[0]):

        target = target_classes[i]
        n_pos = torch.sum(hit[:,target])
        n_gt = torch.sum(test_label[:,target])
        per_class_accuracies[i] = torch.div(n_pos.float(),n_gt.float())
        #pdb.set_trace()
    return per_class_accuracies.mean().item()


def eval_zs_gzsl_k(k,dataloader,model,device,bias_seen,bias_unseen,is_detect=False):
    model.eval()
    print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = 100
    n_classes = dataloader.ntrain_class+dataloader.ntest_class
    in_package = {'model':model,'device':device, 'batch_size':batch_size,'num_class':n_classes}
    
    if is_detect:
        print("Measure novelty detection k: {}".format(k))
        
        detection_mask = torch.zeros((n_classes,n_classes)).long().to(dataloader.device)
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[seenclasses]=1
        detection_mask[seenclasses,:] = detect_label
        
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[unseenclasses]=1
        detection_mask[unseenclasses,:]=detect_label
        in_package["detection_mask"]=detection_mask
    
    with torch.no_grad():
        acc_seen = val_gzsl_k(k,test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen,is_detect=is_detect)
        acc_novel,acc_zs = val_zs_gzsl_k(k,test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen,is_detect=is_detect)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs

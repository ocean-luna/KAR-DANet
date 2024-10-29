# 二分类 混淆矩阵 ROC
import sys
sys.path.append(r"E:\pythonProject1\efficient_kan")
from src.efficient_kan import KAN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 set_device=None,
                 num_classes=1000,
                 include_top=False,
                 include_top_kan=True,
                 groups=1,
                 width_per_group=64):
        super().__init__()
        self.include_top = include_top
        self.include_top_kan = include_top_kan
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * block.expansion, num_classes)
            )
        if self.include_top_kan:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.kan = KAN([512 * block.expansion, 64, num_classes])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.device = set_device
        self.to(self.device)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(f"Shape before avgpool: {x.shape}")
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        if self.include_top_kan:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            print(f"Shape before passing to KAN: {x.shape}")
            x = self.kan(x)
        return x

    def print_layers(self):
        print("ResNet Layers:")
        print(f"Conv1: Output channels = {self.in_channel}")
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            print(f"Layer {idx + 1}:")
            for block in layer:
                print(f"  Block: {block.__class__.__name__}")
                print(f"    Conv1: in_channels = {block.conv1.in_channels}, out_channels = {block.conv1.out_channels}")
                print(f"    Conv2: in_channels = {block.conv2.in_channels}, out_channels = {block.conv2.out_channels}")
                if isinstance(block, Bottleneck):
                    print(
                        f"    Conv3: in_channels = {block.conv3.in_channels}, out_channels = {block.conv3.out_channels}")


def resnet_kan34(set_device, num_classes=1000, include_top=False, include_top_kan=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], set_device=set_device, num_classes=num_classes, include_top=include_top, include_top_kan=include_top_kan)
# 第一个数据集
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建训练数据集
trainset = torchvision.datasets.ImageFolder(root=r"E:\data\pathological-findings\train", transform=transform)
valset = torchvision.datasets.ImageFolder(root=r"E:\data\pathological-findings\val", transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = 7
model = resnet_kan34(set_device=device, num_classes=num_class,
                     include_top=False,
                     include_top_kan=True).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1_scores = []
y_true_all = []
y_pred_all = []
y_prob_all = []

for epoch in range(100):
    print(f"Epoch {epoch + 1}/100")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_pred_all = []
    train_true_all = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), ncols=100):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_true_all.extend(labels.cpu().numpy())
        train_pred_all.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(trainloader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_true_all = []
    val_pred_all = []
    val_prob_all = []
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            val_true_all.extend(labels.cpu().numpy())
            val_pred_all.extend(predicted.cpu().numpy())
            val_prob_all.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss /= len(valloader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 多分类任务下的精度、召回率和F1分数
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_true_all, val_pred_all, average='macro')
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1_scores.append(val_f1)
    print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

    y_true_all.extend(val_true_all)
    y_pred_all.extend(val_pred_all)
    y_prob_all.extend(val_prob_all)

    scheduler.step()

# 绘制训练和验证损失曲线
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# 绘制训练和验证精度曲线
plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()

# 绘制混淆矩阵，并将颜色范围设为0-1
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_all, y_pred_all)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 对混淆矩阵进行归一化

# 使用归一化后的混淆矩阵绘图
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=['polyps', 'ucg0-1', 'ucg-1', 'ucg1-2', 'ucg2', 'ucg2-3', 'ucg3'],
            yticklabels=['polyps', 'ucg0-1', 'ucg-1', 'ucg1-2', 'ucg2', 'ucg2-3', 'ucg3'], vmin=0, vmax=1)

plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# 绘制多分类的ROC曲线
categories = ['polyps', 'ucg0-1', 'ucg-1', 'ucg1-2', 'ucg2', 'ucg2-3', 'ucg3']
y_true_binarized = label_binarize(y_true_all, classes=list(range(num_class)))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_class):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], np.array(y_prob_all)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算宏平均和微平均
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), np.array(y_prob_all).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_class

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})')
plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4, label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})')

colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown']
for i, color in zip(range(num_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {categories[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
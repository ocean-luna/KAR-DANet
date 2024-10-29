import sys
sys.path.append(r"E:\pythonProject1\resnet_kan")
from src.efficient_kan import KAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import torch.autograd
torch.autograd.set_detect_anomaly(True)
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(out_channel)  # 第一个批归一化层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(out_channel)  # 第二个批归一化层
        self.downsample = downsample  # 下采样

    def forward(self, x):
        identity = x  # 保留输入
        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)  # ReLU激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # 残差连接
        out = F.relu(out, inplace=True)  # ReLU激活函数
        return out


class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(width)  # 第一个批归一化层
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(width)  # 第二个批归一化层
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # 第三个卷积层
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # 第三个批归一化层
        self.downsample = downsample  # 下采样

    def forward(self, x):
        identity = x  # 保留输入
        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)  # ReLU激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)  # ReLU激活函数
        out = self.conv3(out)
        out = self.bn3(out)
        out = out+identity  # 残差连接
        out = F.relu(out, inplace=True)  # ReLU激活函数
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
        self.include_top = include_top  # 是否包含顶部全连接层
        self.include_top_kan = include_top_kan  # 是否包含顶部KAN层
        self.in_channel = 64  # 输入通道数
        self.groups = groups  # 组数
        self.width_per_group = width_per_group  # 每个组的通道数
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)  # 输入卷积层
        self.bn1 = nn.BatchNorm2d(self.in_channel)  # 输入批归一化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 构建layer1
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 构建layer2
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 构建layer3
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 构建layer4
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化层
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层
        if self.include_top_kan:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化层
            self.kan = KAN([512 * block.expansion, 64, num_classes])  # KAN层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 初始化卷积层参数

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))  # 下采样层
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))  # 添加块
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, num_classes: int = 1000):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.maxpool(x)  # 最大池化
        x = self.layer1(x)  # 第1个残差模块
        x = self.layer2(x)  # 第2个残差模块
        x = self.layer3(x)  # 第3个残差模块
        x = self.layer4(x)  # 第4个残差模块

        if self.include_top:
            x = self.avgpool(x)  # 平均池化
            x = torch.flatten(x, 1)  # 展平
            x = self.fc(x)  # 全连接层
        elif self.include_top_kan:
            x = self.avgpool(x)  # 平均池化
            x = torch.flatten(x, 1)  # 展平
            x = self.kan(x)  # KAN层
            assert x.dim() == 2 and x.size(1) == num_classes

        return x

    def get_kan_output(self, x):
        return self.kan(x)

    def grad_cam(self, images, target_layer, num_classes):
        # 梯度权重热力图

        def forward_hook(module, input, output):
            self.feature_maps = output  # 保存特征图

        # 确保目标层正确
        target_layer = getattr(self, target_layer, None)
        if target_layer is None:
            print(f"Error: No such layer {target_layer}")
            return None

        # 注册前向钩子
        hook = target_layer.register_forward_hook(forward_hook)

        self.eval()
        output = self.forward(images, num_classes)
        hook.remove()

        # 确保feature_maps已设置
        if not hasattr(self, 'feature_maps'):
            print("Error: feature_maps not set.")
            return None

        # 在前向传播后立即调用retain_grad来保留特征图的梯度
        self.feature_maps.retain_grad()

        heatmaps = []

        for i in range(output.size(0)):
            self.zero_grad()
            output[i, torch.argmax(output[i])].backward(retain_graph=True)

            if self.feature_maps.grad is None:
                print(f"No gradients for sample {i}")
                heatmaps.append(None)
                continue

            grads = self.feature_maps.grad[i]
            pooled_grads = torch.mean(grads, dim=(1, 2))
            for j in range(self.feature_maps.shape[1]):
                self.feature_maps[i, j, :, :] = self.feature_maps[i, j, :, :] * pooled_grads[j]

            heatmap = torch.mean(self.feature_maps[i], dim=0).squeeze()
            heatmap = torch.relu(heatmap)
            heatmap /= torch.max(heatmap)
            heatmaps.append(heatmap.cpu().detach().numpy())

        return heatmaps


def resnet34(set_device, num_classes=1000, include_top=False, include_top_kan=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], set_device=set_device, num_classes=num_classes, include_top=include_top,
                  include_top_kan=include_top_kan)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

trainset = torchvision.datasets.ImageFolder(root=r"E:\data\pathological-findings\train", transform=transform)  # 训练数据集
valset = torchvision.datasets.ImageFolder(root=r"E:\data\pathological-findings\val", transform=transform)  # 验证数据集

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)  # 训练数据加载器
valloader = DataLoader(valset, batch_size=64, shuffle=False)  # 验证数据加载器

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
num_class = 7  # 类别数

model = resnet34(set_device=device, num_classes=num_class, include_top=False, include_top_kan=True).to(device)  # 模型

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # 学习率调度器
criterion = nn.CrossEntropyLoss()  # 损失函数

EPOCHS = 2  # 训练周期数

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")  # 进度条
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, num_classes=num_class)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss + loss.item()
        _, predicted = torch.max(outputs, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        pbar.set_postfix({'loss': running_loss / (total / labels.size(0)), 'accuracy': correct / total})
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {correct / total:.4f}")

model.eval()
val_iter = iter(valloader)
images, labels = next(val_iter)
images, labels = images.to(device), labels.to(device)
outputs = model(images, num_classes=num_class)

# 获取第一个样本的热图
heatmaps = model.grad_cam(images, target_layer='layer4', num_classes=num_class)  # 使用Grad-CAM获取热图
heatmap = heatmaps[0]  # 获取第一个样本的热图

image = images[0].cpu().numpy().transpose((1, 2, 0))  # 转换图像
image = (image - image.min()) / (image.max() - image.min())  # 归一化
if heatmap is not None:
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 调整热图大小
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 使用热图生成彩色映射
    superimposed_img = heatmap * 0.4 + image  # 叠加热图和原始图像
    plt.imshow(superimposed_img)  # 显示叠加后的图像
    plt.axis('off')  # 关闭坐标轴
    plt.show()
else:
    print("Unable to generate heatmap for this sample.")  # 无法生成热图

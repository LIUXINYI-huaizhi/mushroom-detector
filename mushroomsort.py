import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集
data_dir = 'mushroomgra'  # 替换为您的根文件夹路径
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 检查类别
class_names = full_dataset.classes
print(f"Class names: {class_names}")


def visualize_dataset(dataset, num_samples=8, num_rows=2):
    """
    可视化数据集中的样本 (修复通道问题)
    """
    plt.figure(figsize=(15, 5 * num_rows))

    # 创建子图网格
    num_cols = num_samples // num_rows
    if num_samples % num_rows != 0:
        num_cols += 1

    for i in range(num_samples):
        # 获取原始图像和标签
        img_tensor, label = dataset[i]

        # 反标准化处理
        img = img_tensor.numpy().transpose((1, 2, 0))  # 从 (C, H, W) 转为 (H, W, C)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean  # 反标准化
        img = np.clip(img, 0, 1)  # 限制像素值范围

        # 创建子图
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(f"{class_names[label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 可视化训练集中的8个样本，排列成2行4列
visualize_dataset(train_dataset, num_samples=8, num_rows=2)

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 定义模型
class MushroomClassifier(nn.Module):
    def __init__(self):
        super(MushroomClassifier, self).__init__()
        # 使用ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1  # 或使用DEFAULT
        self.model = resnet50(weights=weights)

        # 冻结所有卷积层参数（可选）
        for param in self.model.parameters():
            param.requires_grad = False

        # 全连接层（适配ResNet50）
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # 增大中间层维度
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MushroomClassifier()

import torch.optim as optim
from torch.optim import lr_scheduler

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam([
    {'params': model.model.fc.parameters(), 'lr': 0.001},  # 分类头高学习率
    {'params': model.model.layer4.parameters(), 'lr': 0.0001}  # 解冻最后一部分卷积层
], weight_decay=1e-5)

# 学习率调度器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'model_state_dict': model.cpu().state_dict(),
                    'class_names': class_names,
                    'transform_mean': [0.485, 0.456, 0.406],
                    'transform_std': [0.229, 0.224, 0.225],
                    'input_size': 224
                }, 'mushroom_resnet50.pth')

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # 绘制训练过程的损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Loss.png")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Acc.png")


    plt.show()

    return model

# 开始训练
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

from PIL import Image

# 预测函数
def predict_image(image_path, model, class_names):
    # 加载并预处理图像
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0

    print(f'Predicted: {class_names[pred]} (Probability: {prob:.4f})')
    return pred, prob

# 示例使用
predict_image('mushroomgra\poisonous mushroom sporocarp\cv (1).jpeg', model, class_names)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
from dataset import FlowerDataset
from tqdm import tqdm

# 设置参数
DATA_DIR = "./flower_dataset/"  # 数据集路径
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 14  # 假设花朵数据集有5个类别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = FlowerDataset("./flower_dataset/val.csv", transform=val_transforms)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

data_loaders = {"val": val_loader}
data_sizes = {"val": len(val_dataset)}

# 加载预训练的 ResNet 模型
model = models.resnet18(pretrained=True)

# 替换最后一层为适配分类任务的层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# 训练和验证函数
def val_model(model, best_model_wts):
    best_model_wts = torch.load(best_model_wts)
    model.load_state_dict(best_model_wts)
    model.eval()
    running_corrects = 0
    for inputs, labels in tqdm(data_loaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / data_sizes['val']

    print(f'Val Acc: {epoch_acc:.4f}')


val_model(model, 'resnet_flower_classification.pth')

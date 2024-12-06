import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# Tensorbaord日志
path_log = Path(f"./logs/{time.strftime('%Y%m%d-%H%M%S')}")
writer = SummaryWriter(path_log)

# 检查是否有可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# 数据增强和预处理
transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),      # 随机裁剪
  transforms.RandomHorizontalFlip(),      # 随机水平翻转
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
  def __init__(self, in_ch, out_ch, kernel, stride, padding):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
    self.bn = nn.BatchNorm2d(out_ch)
    self.mish = nn.Mish()
  
  def forward(self, x):
    return self.mish(self.bn(self.conv(x)))

# 定义 CNN 模型
class Model(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.backbone = nn.Sequential(
      CNN(3, 64, 3, 1, 1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      CNN(64, 128, 3, 1, 1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      CNN(128, 256, 3, 1, 1),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.head = nn.Sequential(
      nn.Linear(256 * 4 * 4, 512),
      nn.Mish(),
      nn.Linear(512, num_classes),
    )

  def forward(self, x):
    x = self.backbone(x)
    x = nn.Flatten()(x)
    x = self.head(x)
    return x

# 初始化模型、损失函数和优化器
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
global_step = 0
best_eval_acc = 0

# 训练模型
for epoch in range(num_epochs):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, target)
    _, predicted = torch.max(outputs, 1)
    acc = (target == predicted).sum().item() / batch_size

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    global_step += 1

    if (batch_idx + 1) % 100 == 0:
      writer.add_scalar("chart/loss", loss.item(), global_step)
      writer.add_scalar("chart/train_acc", acc, global_step)
      print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc:.4f}")

  # 测试模型
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      outputs = model(data)
      _, predicted = torch.max(outputs.data, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()

  eval_acc = correct / total
  print(f"Test Accuracy: {100 * eval_acc:.2f}%")
  writer.add_scalar("chart/eval_acc", eval_acc, global_step)

  if eval_acc > best_eval_acc:
    best_eval_acc = eval_acc
    # 保存最优eval模型
    path_save_model = f"cifar10_cnn_model_best_eval.pth"
    torch.save(model.state_dict(), path_log / path_save_model)
    print(f"Best eval model ({100*eval_acc:.2f}%) saved as {path_log / path_save_model}")

# 保存模型
path_save_model = f"cifar10_cnn_model_{global_step}.pth"
torch.save(model.state_dict(), path_log / path_save_model)
print(f"Last model saved as {path_log / path_save_model}")
print(f"Best eval accuracy {100 * best_eval_acc:.2f}%")

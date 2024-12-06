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

# 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据加载和预处理
transform = transforms.Compose([
  transforms.ToTensor(),        # 转换为 Tensor
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义全连接神经网络
class FullyConnectedNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(FullyConnectedNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)  # 输入到隐藏层
    self.relu = nn.ReLU()             # 激活函数
    self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层到输出层

  def forward(self, x):
    x = x.view(x.size(0), -1)  # 展平
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

# 模型实例化
input_size = 32 * 32 * 3  # CIFAR-10 图像大小 (32x32x3)
hidden_size = 256     # 隐藏层神经元数
num_classes = 10      # CIFAR-10 分类数
model = FullyConnectedNN(input_size, hidden_size, num_classes).to(device)

# 定义损失函数和优化器
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
    path_save_model = f"cifar10_fc_model_best_eval.pth"
    torch.save(model.state_dict(), path_log / path_save_model)
    print(f"Best eval model ({100*eval_acc:.2f}%) saved as {path_log / path_save_model}")
    

# 保存模型
path_save_model = f"cifar10_fc_model_{global_step}.pth"
torch.save(model.state_dict(), path_log / path_save_model)
print(f"Last model saved as {path_log / path_save_model}")
print(f"Best eval accuracy {100 * best_eval_acc:.2f}%")
# epoch11 52.54%
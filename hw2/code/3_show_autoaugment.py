import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torchvision import datasets
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

PATH_ROOT = Path(__file__).parent
path_figures = PATH_ROOT / "figures"
path_figures.mkdir(exist_ok=True)

transform = AutoAugment(AutoAugmentPolicy.CIFAR10)

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

def grid_augment():
  train_iter = iter(train_dataset)
  r = 6
  c = 10
  figure, axs = plt.subplots(r, c, figsize=(10, 6))

  for i in range(r):
    for j in range(c):
      img = next(train_iter)[0]
      ax: Axes = axs[i,j]
      ax.set_axis_off()
      ax.imshow(img)
  plt.tight_layout()
  plt.savefig(path_figures / "grid_augment.png", dpi=100)
  plt.show()

def single_augment():
  test_iter = iter(test_dataset)
  img = next(test_iter)[0]
  r = 3
  c = 10
  figure, axs = plt.subplots(r, c, figsize=(10, 3))

  for i in range(r):
    for j in range(c):
      tmp = img if i==j==0 else transform(img)
      ax: Axes = axs[i, j]
      ax.set_axis_off()
      ax.imshow(tmp)
  plt.tight_layout()
  plt.savefig(path_figures / "single_augment.png", dpi=100)
  plt.show()

# grid_augment()
single_augment()

import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./dataset", split="train", transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16()
vgg16_true = torchvision.models.vgg16(weights='IMAGENET1K_V1') # 将会下载 vgg16神经网络 并且 其参数是用 ImageNet数据集 训练好的
print(vgg16_true)

dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 改动现有的网络
# 在vgg16_true中 添加 linear层
# vgg16_true.add_module("vgg16_end_add_linear", nn.Linear(1000, 10))

# 在vgg16_true中的classifier模块 中添加 linear层
vgg16_true.classifier.add_module("vgg16_end_add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 在vgg16_false中的classifier模块的 第7层 修改 linear层参数
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
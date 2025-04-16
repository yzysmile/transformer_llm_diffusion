import torch
import torchvision.models
from torch import nn
from model_save import *

# 加载模型的方式1 -》保存模型的方式1
model = torch.load("vgg16_save_method1.pth")
# print(model)

# 方式2 加载模型
 # 加载模型结构
vgg16 = torchvision.models.vgg16()

 # 加载模型参数
model_para = torch.load("vgg16_save_method2.pth")

 # 将加载的 模型参数 输入到 模型结构 中
vgg16.load_state_dict(model_para)
print(vgg16)
stop = 1

# 方式1加载的缺陷， 加载自定义网络时，必须写出对应的class
# 解决方式： from model_save import *
# 或者 写出对应的class
# class YZY(nn.Module):
#     def __init__(self):
#         super(YZY, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

self_model = torch.load('yzy_method.pth')
print(self_model)
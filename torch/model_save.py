import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()

# 保存方式1,保存 模型结构 和 模型参数
torch.save(vgg16, "vgg16_save_method1.pth")

# 保存方式2, 仅保存 模型参数 到 字典中（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_save_method2.pth")

# 保存方式2, 仅保存 模型参数 到 字典中（官方推荐）
torch.save({'state_dict':vgg16.state_dict()}, "model.tar")

# 方式1的缺陷
class YZY(nn.Module):
    def __init__(self):
        super(YZY, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

yzy = YZY()
torch.save(yzy, "yzy_method.pth")
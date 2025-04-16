import torch
from torch import nn
# from torch.nn import MaxPool2d # torch.nn 中包含了 多个类
import torchvision # torchvision 是一个库

from torch.utils.data import DataLoader # DataLoader 本身是 一个类
from torch.utils.tensorboard import SummaryWriter # SummaryWriter 本身是 一个类

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (1, 5, 5)) # (N C H W) 或 （C H W）
#
# print(input.shape)

# 池化层的目的是 保留数据特征的前提下 进行数据缩小
class yzy(nn.Module):
    def __init__(self):
        super(yzy, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

yzy = yzy()
# output = yzy(input)
# print(output)

writer = SummaryWriter("log_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images("input", imgs, step)
    output = yzy(imgs) # imgs是(N C H W)格式， [64， 3， 32， 32]
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
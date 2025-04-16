import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter # SummaryWriter 本身是 一个类

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = torch.utils.data.DataLoader(dataset, 64, shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter("log_sigmoid")

class yzy(nn.Module):
    def __init__(self):
        super(yzy, self).__init__()
        self.relu1 = nn.ReLU(inplace=False) # 参数inplace 设置为 False时（不替换） ，可保留原始数据
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


yzy = yzy()

step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, step)
    output = yzy(imgs)
    writer.add_images("sigmoid", output, step)
    step = step + 1

writer.close()

# output = yzy(input)
# print(output)

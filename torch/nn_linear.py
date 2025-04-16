import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)


class YZY(nn.Module):
    def __init__(self):
        super(YZY, self).__init__()
        self.linear1 = nn.Linear(in_features=196608, out_features=10)

    def forward(self, input):
        output = self.linear1(input)
        return output

yzy = YZY()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = yzy(output)
    print(output.shape)
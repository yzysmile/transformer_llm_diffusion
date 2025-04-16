import torch
from torch import nn


class YZY(nn.Module):
    def __init__(self):
        super(YZY, self).__init__()  # 用于调用 父类构造函数的 方法

    def forward(self, input):
        output = input + 1
        return output


yzy = YZY()
x = torch.tensor(1.0)
y = yzy(x)
print(y)
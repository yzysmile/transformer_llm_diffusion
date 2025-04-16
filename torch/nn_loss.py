import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
print(inputs.shape)
# inputs = torch.reshape(inputs, [1, 1, 1, 3])

targets = torch.tensor([2, 3, 4], dtype=torch.float32)
# targets = torch.reshape(targets, [1, 1, 1, 3])

loss = nn.L1Loss(reduction="sum") # mean or sum
result = loss(inputs, targets)
print(result)

x = torch.tensor([0.1, 0.2, 0.3]) # 预测为 种类1 种类2 种类3 的概率分别为 0.1， 0.2， 0.3
x = torch.reshape(x, [1, 3]) # batchsize为1， 种类为3

y = torch.tensor([1]) # 真实为种类2

# 交叉熵多用于 分类任务
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
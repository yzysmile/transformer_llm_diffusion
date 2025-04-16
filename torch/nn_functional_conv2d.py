import torch
import torch.nn.functional as F

# tensor类型的 二维数组
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1],
                      [1, 2, 1, 0, 0]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 区分torch.nn.functional.cond2d 和 torch.nn.Conv2d
# 前者是 一个函数，后者是 一个类
output = F.conv2d(input, kernel, stride=1)
print(output)

output1 = F.conv2d(input, kernel, stride=1, padding=[1, 1]) # ctrl+p 可以提醒参数
print(output1)

import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F

#
I = torch.eye(3)[None, :, :]

# 创建原始数组
trans = torch.randn(2, 3, 3)
loss = torch.norm(trans, dim=(1, 2))

# 提取三维张量 x 的各个维度
# B D N 分别表示 tensor的 第一维度：有几个 (3-1)维 张量   第二维度：有几个(3-2)维 张量     第三维度：一维张量由的元素个数
#                             batchsize        每一个batch中有多少个点     每一个点由多少个channel构成（包含x y z x' y' z'）
# B, D, N = x.size() # 2 1 3

# Dataloader中比较 使用worker_init_fn 与 不使用使用worker_init_fn 的区别

# 创建一个简单的数据集类
# 简单的数据集，其中包含数字 0 到 9，您想要使用 DataLoader 来加载这些数据进行训练。
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 创建数据集，包含数字 0 到 9
data = list(range(10))
dataset = SimpleDataset(data)

# 不使用 worker_init_fn，数据加载的随机性不可复现
def load_data_without_worker_init_fn():
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(4):
        print(f"Epoch {epoch + 1}:")

        # batch_idx 和 batch 分别用于接受 enumerate(dataloader) 返回的 batch的 index和data;
        for batch_idx, (batch) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            # batch_idx 是批次的索引
            # batch 是数据加载器返回的数据批次
            print(f"Batch {batch_idx}: {batch}")



# 使用 worker_init_fn，数据加载的随机性可复现
def worker_init_fn(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def load_data_with_worker_init_fn():
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,

    )

    for epoch in range(4):
        print(f"Epoch {epoch + 1}:")

        for batch in dataloader:
            print(batch)


print("Without worker_init_fn:")
load_data_without_worker_init_fn()

print("\nWith worker_init_fn:")
load_data_with_worker_init_fn()






test_data = torchvision.datasets.CIFAR10("./dataset", False, transform=torchvision.transforms.ToTensor())

# shuffle为true时， 每一次epoch 的 dataset 进行打乱
# DataLoader是一个数据加载器，用于从 Dataset 中按批次（batch）加载数据。它的主要作用是 管理数据加载过程。
test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=4, drop_last=False)

# 测试集中的第一张图片 及 target
img, target = test_data[0]
print(img.shape) # ([3, 32, 32])； 单张 三通道， 32*32
print(target) # 3; 该张标签为3

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape) # （[4, 3, 32, 32]）； 四张 三通道， 32*32
        # print(targets) # [4, 3, 32, 31] ; 四张 标签 分别为 4，3，32，31
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()


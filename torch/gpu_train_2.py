import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time # 用于比较CPU 和 GPU 用于训练的时间

# from model import *

# 利用GPU进行网络训练 只需要对 网络模型对象、DataLoader加载的 数据以及标签、损失函数 调用其 cuda()方法 或 to(device)即可

# 定义训练的设备: 在第一张显卡上进行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练数据集
train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
print("训练数据集的长度为：{}".format(train_data_size)) # 字符串格式化，将{} 替换为 train_data_size

# 测试数据集
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data_size = len(test_data)
print("测试数据集的长度为：{}".format(test_data_size)) # 字符串格式化，将{} 替换为 train_data_size

# 利用DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 引入神经网络 from model import *

# 创建网络模型
class YZY(nn.Module):
    def __init__(self):
        super(YZY, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

yzy = YZY()
yzy = yzy.to(device)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器,SGD为随机梯度下降
learning_rate = 1e-2
optimizer = torch.optim.SGD(yzy.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练轮数
epoch = 10

# 可视化
writer = SummaryWriter("train_log")
start_time = time.time()

# 训练
for i in range(epoch):
    print("--------第{}轮训练--------".format(i+1))

    yzy.train() # 非必要,当网络模型中 包含 Dropout, BatchNorm,etc 才是必要的
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 正向传播
        outputs = yzy(imgs)
        loss = loss_fn(outputs, targets) # 计算损失

        optimizer.zero_grad()  # 反向传播前 梯度清零
        loss.backward() # 反向传播

        optimizer.step() # 参数优化更新
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time) # 单位为 秒
            print("训练次数：{}， loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每进行一轮训练后，进行验证
    yzy.eval()  # 非必要,当网络模型中 包含 Dropout, BatchNorm,etc 才是必要的
    total_test_loss = 0
    total_accuracy = 0 # 分类问题，统计正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = yzy(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss.item()))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(yzy, "yzy_{}.pth".format(i))
    # 或者 torch.save(yzy.state_dict(), "yzy_{}.pth".format(i))
    print("第{}轮训练的模型已保存".format(i))

writer.close()
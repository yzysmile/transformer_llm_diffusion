import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=25),
                                                     torchvision.transforms.ToTensor()])

# 将datasets中的每一张 PIL Image 转换为 Tensor数据类型
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transforms, download=True)

# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# img.show()

writer = SummaryWriter("p10") # 日志名称为"p10"
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
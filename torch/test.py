import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./imgs/dog.png"
image = Image.open(image_path)
image = image.convert("RGB")
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),  # 通过 上、下采样(插值)进行
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

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

# 加载的模型是用GPU训练的
model = torch.load("yzy_9.pth")
image = image.cuda() # 验证的模型是用GPU训练， 输入的数据也用调用其 cuda()方法

# 或者将模型转化到CPU上
# model = torch.load("yzy_9.pth", map_location=torch.device("cpu"))

print(model)
image = torch.reshape(image, (1, 3, 32, 32)) # batch_size， 通道数， 32*32

model.eval()
with torch.no_grad():
    output = model(image)
print(output)

# 打印其预测的类别 0 1 2 3 4 5 6 7 8 9,其中5代表 狗
print(output.argmax(1))
# 加载一个预训练的超分辨率模型 并 执行推理

import os

import cv2
import numpy as np
import requests  # 用于发送HTTP请求，使开发者能够与Web服务进行交互
import torch
from torch import nn

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

# Download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://raw.githubusercontent.com/open-mmlab/mmagic/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)

def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    # torch.load('srcnn.pth') 加载"预训练模型"，其中保存 模型参数、优化器等多种信息，返回一个字典
    # torch.load('srcnn.pth')['state_dict']返回的依旧是 一个字典
    state_dict = torch.load('srcnn.pth')['state_dict']

    keys_list = list(state_dict.keys())

    # 遍历state_dict中的key以去掉前缀，前缀通常是因为模型是在nn.Module的子类中定义并使用了.to(device)或.cuda()的操作
    for old_key in keys_list:
        # old_key.split('.')[1:] 分割old_key字符串，从第二个元素(即第一个'.'之后的部分)开始取值，因为第一个元素是需要去掉的前缀
        # '.'.join(...) 将分割后的部分重新用.连接起来，形成新的键名
        new_key = '.'.join(old_key.split('.')[1:])
        # 更新字典，用新的键名代替旧的键名
        state_dict[new_key] = state_dict.pop(old_key)

    # 加载模型参数
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# 推理
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)


# 中间表示（Intermediate Representation, IR）
# ONNX(Open Neural Network Exchange) 是 用于描述计算图的一种格式，是 深度学习框架 到 推理引擎 的桥梁
# PyTorch自带把模型对象 转成 ONNX格式的函数torch.onnx.export(...)，前三个必选参数：
#  (1) 要转换的模型； （2）模型的任意一组输入； (3)导出的ONNX文件的文件名
# 为模型提供一组输入的目的： PyTorch提供了一种叫做追踪（trace）的模型转换方法：给定一组输入，再实际执行一遍模型，即把这组输入对应的计算图记录下来，保存为 ONNX 格式。

import torch.onnx

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, x, 'srcnn.onnx', opset_version=11, input_names=['input'], output_names=['output'])

# 验证.onnx模型文件是否正确
import onnx

onnx_model = onnx.load("srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print('Model incorrect')
else:
    print('Model correct')

# 使用Netron可视化ONNX模型，可视化的结果是一个计算图
# 推理引擎ONNXRuntime,ONNXRuntime是直接对接ONNX的，即ONNXRuntime可以直接读取并运行".onnx"，而不需再把.onnx格式 转换 成其他格式的文件
# 对于PyTorch - ONNX - ONNXRuntime 这条部署流水线，只要在目标设备中得到 .onnx 文件，并在 ONNXRuntime 上运行模型，模型部署就算大功告成

import onnxruntime

# onnxruntime.InferenceSession 用于获取一个用于 “srcnn.onnx” 的ONNXRuntime推理器
ort_session = onnxruntime.InferenceSession("srcnn.onnx")  # 'ort' 是 ONNXRuntime的缩写

# 推理器对的run方法 用于 模型推理，run(...) 的接受参数分别是 输出张量名的列表、输入值的字典 并 返回一个 list
ort_output = ort_session.run(['output'], {'input': input_img})[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite('face_ort.png', ort_output)

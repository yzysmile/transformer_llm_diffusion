# 在此以之前的 超分辨模型SRCNN为例，之前的例子中，我们用ONNXRuntime作为后端，通过PyTorch的 symbolic函数 导出了一个支持 动态scale的ONNX模型，该模型可直接用ONNXRuntime运行
# 这是因为 NewInterpolate类 导出的节点Resize 就是ONNXRuntime支持的节点， 在此直接将 srcnn3.onnx 转换到 TensorRT

# 安装mmdeploy后，确保安装了mmcv

from mmdeploy.backend.tensorrt.utils import from_onnx
import os

# 直接调用from_onnx的转换方法 与 chapter6的01_.py的转换方法 的主要区别在于 易用性和控制程度
 # 1. chapter6的01_.py的转换方法 高度定制化的构建过程或对转换过程中的每个细节都有精确的控制需求
 # 2. from_onnx寻求快速简便的转换过程，且不需要深入到每个配置细节，一个封装好的函数如from_onnx， 但使用这类函数时，了解其内部实现和限制是很重要的。
# from_onnx(
#     'srcnn3.onnx',
#     'srcnn3',  # 输出模前缀，转换后的TensorRT模型以此为基础命名，最终生成的文件名可能为srcnn3.trt
#     input_shapes=dict(  # 两个输入分别是 input和factor
#         input=dict(  # min_shape, opt_shape, max_shape 被设置为相同值，意味着输入尺寸是固定的。
#             min_shape=[1, 3, 256, 256],
#             opt_shape=[1, 3, 256, 256],
#             max_shape=[1, 3, 256, 256]),
#         factor=dict(  # 上采样因子，其最小、最优、和最大形状也都一致，表明该输入的尺寸不会变化
#             min_shape=[4],
#             opt_shape=[4],
#             max_shape=[4]))
#     )

#  onnx 转 trt失败有以下两个方面：
#  1. srcnn3.onnx 中的 Resize 是ONNX的原生节点，srcnn3.onnx的插值方式'bicubic'不被TensorRT支持
#     (TensorRT的Resize Layer仅支持nearest和bilinear两种插值方式)
#  2. 但即使将‘bicubic’改为‘bilinear’，转换仍然失败，这是因为TensorRT无法接受动态scale导致
#  体现了chapter2-模型部署中的难题之：中间表示(ONNX) 与 推理引擎（TensorRT）的兼容问题

# 为了解决上述问题，需要创建一个新的ONNX节点 替换为 原生的Resize节点，并且实现新节点对应的插件代码
# 定义新的ONNX算子 名为 Test::DynamicTRTResize

import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np
import os, requests

tensor = torch.randn(1, 3, 256, 256)
shape = tensor.shape
print(type(shape))
shape_last = shape[0]

shape_tensor = tensor.size(0)
print(type(shape_tensor))

# Download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmagic/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)

class DynamicTRTResize(torch.autograd.Function):
    def __init__(self)->None:
        super().__init__()

    def symbolic(g, input, size_tensor, align_corners = False):
        """Symbolic function for creating onnx op."""
        return g.op(
            'Test::DynamicTRTResize',  # 自定义的ONNX算子
            input,
            size_tensor,
            align_corner_i=align_corners
        )

    def forward(g, input, size_tensor, align_corners = False):  # 这里的第一个参数不应该是"ctx"吗？
        size = [size_tensor.size(-2), size_tensor.size(-1)]  # .size() 和 .shape 返回包含张量维度信息的元组(tuple)，（-2）获取倒数第二个维度大小(width)
        # size = 512
        # 对input进行 上(下)采样 的方法
        # torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        # 上（下）采样后的size既可以指定为 目标size 也可以指定为 scale_factor，
        # size (int or (int) or [int, int] or [int, int, int]) – output spatial size.
        # Tensor interpolated to either the given size or the given scale_factor
        return interpolate(input, size=size, mode='bicubic', align_corners=align_corners)  # interpolate是PyTorch中的方法(算子)


class StrangeSuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, size_tensor):
        x = DynamicTRTResize.apply(x, size_tensor)  # 调用自定义的onnx算子 并给予输入，该算子的作用是 缩放输入张量；是调用DynamicTRTResize类中的forward函数
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

def init_torch_model():
    torch_model = StrangeSuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
factor = torch.rand([1, 1, 512, 512], dtype=torch.float)

input_img = cv2.imread('face.png').astype(np.float32)  # ndarray:(256, 256, 3)

# HWC to CHW ( ndarray(256, 256, 3) -> ndarray(3, 256, 256) )
input_img = np.transpose(input_img, [2, 0, 1])

# CHW to NCHW ( ndarray(3, 256, 256) ->  ndarray(1, 3, 256, 256))
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Save image
cv2.imwrite("face_torch.png", torch_output)

# ONNX模型导出
x = torch.randn(1, 3, 256, 256)

dynamic_axes = {
    'input': {
        0: 'batch',
        2: 'height',
        3: 'width'
    },
    'factor': {
        0: 'batch1',
        2: 'height1',
        3: 'width1'
    },
    'output': {
        0: 'batch2',
        2: 'height2',
        3: 'width2'
    },
}

with torch.no_grad():
    torch.onnx.export(
       model, (x, factor), 'srcnn3.onnx',
       opset_version=11,
       input_names=['input', 'factor'],
       output_names=['output'],
       dynamic_axes=dynamic_axes
    )

# 直接将该模型转换成TensorRT模型是不行的，因为TensorRT无法解析 DynamicTRTResize 节点
# MMDeploy中实现了 Bicubic Interpolate算子，故可以复用CUDA部分代码，只针对TensorRT实现支持动态的scale的插件即可
# 该部分代码用C++实现
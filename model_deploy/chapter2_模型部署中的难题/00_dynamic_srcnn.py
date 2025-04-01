# 在chapter1中的SRCNN中，图像的放大比列是固定的，使用upscale_factor控制模型的放大比列，默认值为3，生成了一个放大3倍的PyTorch模型，再转换哪位ONNX格式的模型
# 如果需要一个放大4倍的模型，需要重新生成一遍模型，再做一次到ONNX的转换

# 假设要做一个超分辨率的应用，希望图片的放大倍数能够自由设置，并且不必重新生成.onnx文件
# 因此，必须修改原来的模型，令模型的放大倍数变成推理时的输入

import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        # SuperResolutionNet 未修改之前，nn.Upsample 在初始化阶段固化了放大倍数，
        # PyTorch 的 interpolate 插值算子可以在运行阶段选择放大倍数,从而让模型支持动态放大倍数的超分
        x = interpolate(x, scale_factor=upscale_factor.item(), mode='bicubic', align_corners=False)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

def init_torch_model():
    torch_model = SuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), torch.tensor(3)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch_2.png", torch_output)

# 导出ONNX模型, PyTorch模型再导出到ONNX模型时，模型的输入参数的类型必须全部是torch.Tensor
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, (x, torch.tensor(3)), "srcnn2.onnx", opset_version=11,
                      input_names=['input', 'factor'], output_names=['output'])

# 导出ONNX时报了一条TraceWarning的警告
# 可视化srcnn2.onnx后可以发现 虽然我们把推理模型输入设置为2个，但可视化后的结果仍然只有一个输入
# 这是因为使用了 torch.Tensor.item() 把数据从Tensor里取出来，但导出ONNX模型时 这个操作是无法被记录的

# 直接修改原来的模型似乎行不通，需从 PyTorch 转 ONNX 的原理入手
# 即PyTorch 转 ONNX，实际上就是把每个 PyTorch 的操作映射成了 ONNX 定义的算子
# 现有实现插值的 PyTorch 算子有一套规定好的映射到 ONNX Resize 算子的方法，这些映射出的Resize算子的scales只能是常量
# 需自定义一个实现插值的PyTorch算子，然后将其映射到ONNX Resize算子上
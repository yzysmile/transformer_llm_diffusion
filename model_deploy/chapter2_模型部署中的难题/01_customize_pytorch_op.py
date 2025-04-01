# 00_dynamic_srcnn.py 最终导出ONNX时报了一条TraceWarning的警告
# 可视化srcnn2.onnx后可以发现 虽然把推理模型输入设置为2个，但可视化后的结果仍然只有一个输入
# 这是因为使用了 torch.Tensor.item() 把数据从Tensor里取出来，但导出ONNX模型时 这个操作是无法被记录的

# 直接修改原来的模型行不通，需从 PyTorch 转 ONNX 的原理入手
# PyTorch 转 ONNX的原理 实际上就是把每个 PyTorch 的操作映射成了 ONNX 定义的算子
# 现有实现插值的 PyTorch 算子有一套规定好的映射到 ONNX Resize 算子的方法，这些映射出的Resize算子的scales只能是常量
# 需自定义一个实现插值的PyTorch算子，然后将其映射到ONNX Resize算子上

import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np

# 自定义PyTorch算子，在chapter4再详细介绍
class NewInterpolate(torch.autograd.Function):
    def symbolic(g, input, scales):
        return g.op("Resize", input,
                    g.op("Constant", value_t = torch.tensor([], dtype=torch.float32)),
                    scales, coordinate_transformation_mode_s="pytorch_half_pixel", cubic_coeff_a_f=-0.75,
                    mode_s='cubic', nearest_mode_s="floor")

    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input, scale_factor=scales, mode='bicubic', align_corners=False)


class StrangeSuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

def init_torch_model():
    torch_model = StrangeSuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch_3.png", torch_output)

# 导出模型
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, (x, factor),
                      "srcnn3.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])

# 使用ONNXRuntime 推理 onnx模型
import onnxruntime

input_factor = np.array([1, 1, 6, 6], dtype=np.float32)
ort_session = onnxruntime.InferenceSession("srcnn3.onnx")
ort_inputs = {'input': input_img, 'factor': input_factor}
ort_output = ort_session.run(None, ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort_3.png", ort_output)
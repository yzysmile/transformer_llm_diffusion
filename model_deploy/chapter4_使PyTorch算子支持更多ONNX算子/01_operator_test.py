#  在完成了一份自定义算子后，一定要测试一下算子的正确性
#  用 PyTorch 运行一遍原算子，再用ONNX Runtime运行一下 ONNX 算子，最后比对两次的运行结果

import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

model = Model()
input = torch.rand(1, 3, 10, 10)

#  PyTorch模型推理（即将张量传入PyTorch模型对象）并把推理结果转成numpy格式
torch_output = model(input).detach().numpy()

# onnxruntime 推理
# Tensors 是 ONNX 处理数据的基本单位
# 但确实应该指出输入数据应当被组织为numpy.ndarray，numpy数组可以被无缝转换为ONNXRuntime内部处理的tensor格式
sess = onnxruntime.InferenceSession('asinh.onnx')
ort_output = sess.run(['1'], {'0': input.numpy()})[0]  # [0]获取列表中index为“0”的结果

#  使用 np.allclose保证两个结果ndarray误差在一个可允许的范围内
assert np.allclose(torch_output, ort_output)

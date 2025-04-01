#  case1: ATen库中有PyTorch算子的实现，onnx中有对应的onnx算子实现，只是缺少 PyTorch算子 到 onnx算子 的映射
#         只需添加符号函数，它可以看成是PyTorch算子类的一个静态方法，定义一般如下：
#         def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...):
#         其中torch._C.Graph 和 torch._C.Value对应PyTorch的C++实现里的一些类，在此不深究它们的细节，只需知道第一个参数就固定叫g，表示和计算图相关的内容
#         后面的每个参数都是 PyTorch算子的前向推理输入

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asin(x)  # 调用了PyTorch中的“asin”算子


from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):  # g为固定的第一个参数； 后面的参数与ATEN库中的PyTorch的asinh算子实现函数的接受参数相同
    return g.op("Asinh", input)  # g.op函数的接受参数分别为 onnx算子的名字 及 onnx算子的对应的输入（可查阅官方文档）

# PyTorch中的算子依据 符号函数 映射到 ONNX指定版本的算子集中
    # 第一个参数：ATen库中的实现的PyTorch算子名称
    # 第二个参数：符号函数
    # 第三个参数：ONNX算子的“域”，对于普通的ONNX算子，直接填空字符串即可
    # 第四个参数：向第几个版本的ONNX算子注册
# 即
register_op('asinh', asinh_symbolic, '', 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'asinh.onnx')
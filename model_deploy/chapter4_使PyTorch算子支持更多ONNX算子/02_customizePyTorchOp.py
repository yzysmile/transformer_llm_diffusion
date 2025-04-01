#  case2: ATen库中没有PyTorch算子的实现，这时要考虑先自定义一个PyTorch算子，再把它转换到onnx中。
#  新增PyTorch算子 官方的推荐方法是 添加TorchScript算子
#  添加PyTorch算子的方法繁琐，先跳过新增 TorchScript 算子的内容，首先为现有TorchScript算子(以 Deformable Convolution为例)添加onnx支持的方法
#  可变形卷积（Deformable Convolution）是在 Torchvision 中实现的 TorchScript 算子

#  有了ATen库支持的PyTorch算子 经验后，为PyTorch算子添加 符号函数 要经过以下步骤：
#  1.获取 PyTorch算子 的前向推理接口
#  2.获取 对应的ONNX算子的定义
#  3.编写符号函数并绑定

import torch
import torchvision

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

    def forward(self, x):
        return self.conv2(x, self.conv1(x))

#  经查询DeformConv2d层 最终调用 'deform_conv2d' PyTorch算子；再在ONNX官方文档中查找对应算子的定义。
#  去 ONNX 的官方算子页面搜索 “deform”，将搜不出任何内容。目前，ONNX 还没有提供可变形卷积的算子，我们要自己定义一个 ONNX 算子了。
#  转换为ONNX模型是一套标准，故Pytorch转换为ONNX模型 ONNX算子本身可以不包括实现，
#  这里简略定义一个ONNX的 可变形卷积 算子，而不实现ONNX算子的实现

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def symbolic(g,  # 第一个参数固定为'g'
             # 其余参数为 PyTorch算子的 输入
             input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask):
    return g.op("custom::deform_conv2d", input, offset)  # 参数 onnx算子名称 及 该onnx算子的输入

# PyTorch算子 依据符号函数 映射到onnx算子
    # 第一个参数：“PyTorch”中的 算子名
    # 第二个参数：符号函数
    # 第三个参数：onnx算子域（Optional）
    # 第四个参数：onnx算子集版本
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'dcn.onnx')
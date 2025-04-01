#  case3:一种简单添加 PyTorch算子 的实现方法，来代替较为复杂的新增 TorchScript算子，同时使用torch.autograd.Function 封装此个算子。
#  直接使用Python接口 调用 C++函数 更加优雅的方式是把这个 调用接口封装起来，在此使用 torch.autograd.Function 封装PyTorch算子的底层调用

#  使用 torch.autograd.Function以class封装PyTorch算子的底层调用 步骤：
#  1. PyTorch算子类 继承torch.autograd.Function
#  2. 定义 forward函数，参数为 固定的'ctx' 和 PyTorch算子 接受参数
#  3. 定义 symbolic函数（将PyTorch算子 与 onnx算子 映射起来），参数为 固定的'g' 和 PyTorch算子 接受参数
#  使用torch.autograd.Function 不必再调用 torch.onnx.symbolic_registry 中的 register_op方法
import torch
import my_lib  # my_lib路径需放在Python解释器下

#  Function 类本身表示 PyTorch 的一个可导函数
class MyAddFunction(torch.autograd.Function):
    def forward(ctx, a, b):  # ctx即"context"
        return my_lib.my_add(a, b)

    def symbolic(g, a, b):  # g为固定参数，其余参数为 Pytorch算子 的输入
        # g.op() 定义了三个算子：常量、乘法、加法
        const_tensor_two = g.op("Constant", value_t=torch.tensor([2]))  # 常量算子， 指定了常量节点的值为一个张量 [2]
        a = g.op('Mul', a, const_tensor_two)  # 乘法算子
        return g.op('Add', a, b)  # 加法算子

#  my_add算子封装成Function后，不必使用register_op 将PyTorch算子 依据符号函数 映射到 ONNX算子
#  此时直接将 forward方法的实现 与 符号函数 自动绑定

# 使用 Function 的派生类做推理时，不应该显式地调用 forward，而应该调用其 apply 方法
my_add = MyAddFunction.apply

#  使用 torch.autograd.Function 封装PyTorch算子的底层调用后，可进一步把算子封装成一个神经网络中的计算层
#  封装了my_add，就和封装了conv2d 的 torch.nn.Conv2d 一样
class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)

#  “包装”新算子完成后，最后进行onnx模型导出
model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), 'my_add.onnx')
torch_output = model(input, input).detach().numpy()

# 验证onnx模型的正确性
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('my_add.onnx')
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]
assert np.allclose(torch_output, ort_output)


# 上一个脚本中，通过ONNX提供的API，搞懂了ONNX由哪些模块构成，如何读取现有的".onnx"文件并从中提取模型信息

import onnx
# 读取ONNX模型
# 在保存模型时，我们传给onnx.save的是一个 ModelProto 的对象；
# 同理，onnx.load读取ONNX模型时，返回的也是一个ModelProto对象
model = onnx.load('linear_func.onnx')

# 当想得知ONNX模型数据有哪些属性时，只需要把模型输出即可
print(model)

# 得到该对象后，来看看怎么把 图GraphProto、节点NodeProto、张量信息 ValueInfoProto 读取出来
# 构造onnx时是 自下而上(Value->Node->Graph->Model)， 读取onnx是 自上而下(Model->Graph->Node->Value)
graph = model.graph
node = graph.node
input = graph.input
output = graph.output
print('------分割线0------')
print(node)  # node其实是一个列表，列表中包含属性 input, output, op_type
print('------分割线1------')
print(input)
print('------分割线2------')
print(output)

# 获取node列表里的第一个节点'Mul'的属性
node_0 = node[0]
node_0_inputs = node_0.input
node_0_outputs = node_0.output
input_0 = node_0_inputs[0]
input_1 = node_0_inputs[1]
output = node_0_outputs[0]
op_type = node_0.op_type

print('------分割线3------')
print(input_0)
print('------分割线4------')
print(input_1)
print('------分割线5------')
print(output)
print('------分割线6------')
print(op_type)

# 写ONNX模型，包括：
# 1.修改ONNX模型；
# 2.使用ONNX的子模型提取功能，对ONNX模型进行调试；

# 1.修改ONNX模型
# 方式1：按照ONNX模型构造的方法，新建 节点与张量信息，与原有模型组合成一个新的模型
# 方式2：在不违反ONNX规范的前提下，直接修改某个数据对象的属性，在此举例
print('------分割线7------')
print(node[1].op_type)  # 打印原来node[1]的 算子类型 属性
node[1].op_type = 'Sub'  # 修改node[1]的 算子类型 属性

onnx.checker.check_model(model)
onnx.save(model, 'linear_func_2.onnx')
model_2 = onnx.load("linear_func_2.onnx")
print('------分割线8------')
print(model_2.graph.node[1].op_type)  # 修改后的模型的 node[1]的 算子类型 属性

# 2.子模型提取
# 实际部署中，如果用PyTorch导出的ONNX模型出现问题
# 在此之前，一般通过修改PyTorch的代码来解决，而不会从ONNX入手，把ONNX模型当成一个不可修改的黑盒看待
# 但现在学习了ONNX的原理后，可尝试对ONNX模型本身进行调试，在此例举如何巧妙利用ONNX提供的子模型提取(extract)功能，对ONNX模型进行调试
import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3,3,3),
                                          torch.nn.Conv2d(3, 3,3),
                                          torch.nn.Conv2d(3, 3, 3))

        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))

        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))

        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))

    def forward(self, x):
        x = self.convs1(x)
        x1 = self.convs2(x)
        x2 = self.convs3(x)
        x = x1 + x2
        x = self.convs4(x)
        return x

model_PyTorch = Model()
input = torch.randn(1, 3, 20, 20)  # batch_size, channel, h, w

torch.onnx.export(model_PyTorch, input, 'whole_model.onnx')

# ONNX模型的边 是 张量，边是有序号的，由PyTorch自动生成，边的序号 实际是 前一个节点的输出张量 和 后一个节点的输入张量 序号
# 张量(ONNX模型的边)的序号，可以在 Netron 中可视化点击边，在其name属性 即 边的序号；边的序号 是否可以 直接在PyTorch中 查看？
# 提取ONNX模型的子模型，使用onnx.utils.extract_model(...)接口时，其参数分别是 原模型、输出子模型、原模型 起始边 和 终点边 的序号
# 2.1.单纯子模型提取
onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['22'], ['28'])  # 22 和 28 是张量(边) 序号

# 2.2.子模型提取时，添加额外输出
onnx.utils.extract_model('whole_model.onnx', 'submodel_1.onnx', ['22'],['27', '31'])  # 在此 额外输出 27号 张量

# 2.3.子模型提取时，添加冗余输入(如果只用子模型的 部分输入就能得到输出，那么那些“较早”的输入就是冗余的)
# e.g. 输入序号为22的张量，即可得到输出为28的张量，故 input.1张量 是 冗余的
onnx.utils.extract_model('whole_model.onnx', 'submodel_2.onnx', ['22', 'input.1'], ['28'])

# 2.4.子模型提取时，输入信息不足，将会报错
# onnx.utils.extract_model('whole_model.onnx', 'submodel_3.onnx', ['24'], ['28'])  # ERROR

# 2.5.输出ONNX模型的中间节点值
# 使用ONNX模型时，最常见的一个需求是能够使用推理引擎输出中间节点的值，这多见于 PyTorch模型(.pth) 和 ONNX模型（.onnx）的精度对齐
# 因为只要能够输出 中间节点的值（即 2.2. 用法），就能定位到精度出现偏差的算子
# 在以下提取的 子模型中，新增了一些输出，在输出序号为‘31’的张量同时，把其他几个 序号的张量 也加入了 输出中
onnx.utils.extract_model('whole_model.onnx', 'more_output_model.onnx', ['input.1'], ['31', '23', '25', '27'])

# 这样的话，用ONNX Runtime运行 more_output_model.onnxs时，就能得到更多的输出了
# 为了方便调试，还可以把原模型拆分成 多个互不相交的 子模型，这样，可以只对原模型的部分子模块调试，e.g.
onnx.utils.extract_model('whole_model.onnx', 'debug_model_1.onnx', ['input1.1'], ['23'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_2.onnx', ['23'], ['25'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_3.onnx', ['23'], ['27'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_4.onnx', ['25', '27'], ['31'])

# 总结：
# 1. onnx.save(...)保存模型，onnx.load(...)读取模型，onnx.checker.check_model(...)可以检查模型是否符合规范
# 2. onnx.utils.extract_model(...) 可从原模型中取出子模型，实现对ONNX模型的调试

# 子模型提取固然是一个有效的ONNX调试工具，但在使用PyTorch等框架导出ONNX模型时，仍然有两个问题：
# 1.一旦PyTorch模型改变，ONNX模型的 张量（边）的序号也会改变。这样每次提取同样的子模块时都要重新去ONNX模型中查序号，如此繁琐的调试方法是不会在实践中采用的。
# 2.难以将 PyTorch代码 和 ONNX节点 对应起来，当模型结构变得十分复杂时，要识别ONNX中每个节点的含义是不可能的。
# MMDeploy为 PyTorch模型 添加了 模型分块功能，使用该功能，能够通过只修改PyTorch模型的实现代码 把 原模型导出成 多个互不相交的子ONNX模型。



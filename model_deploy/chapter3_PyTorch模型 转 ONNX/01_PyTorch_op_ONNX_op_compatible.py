# PyTorch 转 ONNX 时最容易出现的问题就是算子不兼容了，在此介绍如何判断某个 PyTorch算子在 ONNX 中是否兼容
# 转换普通的torch.nn.Module模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；
#                                       另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子
# 在此翻译过程中有以下情况：
# （1） PyTorch算子 一对一翻译成 ONNX算子
# （2） PyTorch算子 会翻译成 多个ONNX算子
# （3） PyTorch算子 没有对应的 ONNX算子，转换报错

# 如何查看PyTorch算子 与 ONNX算子 的对应情况呢？
# 在此先看ONNX算子的定义情况，再看一下PyTorch定义的算子映射关系

# ONNX算子文档
# 官方的算子文档中，可以查看某个算子的输入、输出参数、规定 及 使用实例

# PyTorch算子 对 ONNX算子 的映射
# 在PyTorch中，和 ONNX有关的定义全部放在 torch.onnx目录中
# symbloic_opset{n}.py 即表示 PyTorch 在支持第 n 版 ONNX 算子集时新加入的内容
# e.g. 在torch/onnx 文件夹中搜索'bicubic'，可以发现这个插值在11个版本的定义文件中
#      每一个g.op就是一个 ONNX的定义
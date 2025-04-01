# torch.onnx.export(
#                   model, args, name, export_params=True, verbose=False, training=TrainingMode.EVAL,
#                   input_names=None, output_names=None, ...
#                   )
# 前三个必选参数为 model-模型、args-模型输入、name-导出的ONNX文件名
#
# export_params-模型中是否存储模型权重
# input_names, output_names-设置输入和输出张量的名称, 若不设置，会自动分配一些数字为名字
# opset_version-转换时参考哪个 ONNX 算子集版本
# dynamic_axes-指定输入输出张量的哪些维度是动态的，为追求效率，ONNX 默认所有参与运算的张量的形状不发生改变

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        return x

model = Model()
dummy_input = torch.rand(1, 3, 10, 10)
model_names = ['model_static.onnx', 'model_dynamic_0.onnx', 'model_dynamic_23.onnx']

# 动态维度数据指定 是一个dict
# { '节点名称': {张量的具体某个维度: '当前维度的含义'} }
dynamic_axes_0 = {
    'in': {0: 'batch'},  # 输入张量的第0个维度是动态维度，表示“batch”
    'out': {0: 'batch'}  # 输出张量的第0个维度是动态维度，表示“batch”
}
dynamic_axes_23 = {
    'in': {2: 'width', 3: 'height'},  # 输入张量的第2、3个维度是动态维度，分别表示“width”、“height”
    'out': {2: 'width', 3: 'height'}
}

# 输入、输出没有动态维度
torch.onnx.export(model, dummy_input, model_names[0], input_names=['in'], output_names=['out'])

# 输入、输出在第0维(维度名字为"batch") 为动态维度
torch.onnx.export(model, dummy_input, model_names[1], input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0)

# 输入、输出在第2、3维(维度名字分别为"width","height") 为动态维度
torch.onnx.export(model, dummy_input, model_names[2], input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_23)

# 验证动态维度的作用
import onnxruntime
import numpy as np

origin_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
mult_batch_tensor = np.random.rand(2, 3, 10, 10).astype(np.float32)
big_tensor = np.random.rand(1, 3, 20, 20).astype(np.float32)

inputs = [origin_tensor, mult_batch_tensor, big_tensor]
exceptions = dict()

# 每一个onnx模型 都要输入origin_tensor、mult_batch_tensor、big_tensor
for model_name in model_names:
    for i, inputs in enumerate(inputs):
        try:
            ort_session = onnxruntime.InferenceSession(model_name)
            ort_session.run(['out'], {'in': input})
        except Exception as e:
            exceptions[(i, model_name)] = e
            print(f'Input[{i}] on model {model_name} error.')
        else:
            print(f'Input[{i}] on model {model_name} succeed.')

# 使用python API搭建TensorRT网络
# TensorRT用于高效实现已训练好的深度学习模型的推理过程的SDK，内含“推理优化器”和“运行时环境”，使DL模型能以“搞吞吐”和“低延时”运行；
# 推理优化器（构建期）
 # ONNX模型解析/建立：加载ONNX格式的模型/使用TensorRT原生API搭建模型
 # 计算图优化： 横向层融合（Conv），纵向层（Conv+add+ReLU）融合，...
 # 节点消除： 去除无用层，节点变换（Pad， Slice, Concat, Shuffle），...
 # 多精度支持： FP32/FP16/INT8/TF32（可能插入reformat节点，用于数据类型变换）
 # 优选kernel/format： 硬件有关优化
 # 导入plugin： 实现自定义操作
 # 显存优化： 显存池复用

# 运行期（运行时环境）
 # 运行时环境： 对象生命期管理，内存显存管理，异常处理
 # 序列化、反序列化： 推理引擎保存文件 或 从文件中加载

# 有C++ 和 Python 的API

# 需要注意的是对于 权重部分，如 卷积 或 归一化层，需要将权重内容赋值到 TensorRT网络中，在此不展示，只搭建一个对输入做池化的简单网络

# 使用python API直接搭建TensorRT网络 主要是利用 tensorrt.Builder 的 create_builder_config、create_network 分别搭建 config、network
# config用于设置网络的 最大工作空间等参数，network是网络主体，需要对其逐层添加内容
# 此外，需要定义好 输入 和 输出名称，将构建好的网络序列化，保存成本地文件。

import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    pool = network.add_pooling(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    pool.stride = (2, 2)
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]]*3)
    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))  # 序列化
        print("generating file done!")
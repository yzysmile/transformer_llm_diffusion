import torch
import onnx
import tensorrt as trt

onnx_model = 'model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

device = torch.device('cuda:0')

# 导出.onnx模型
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11)

# 加载.onnx模型
onnx_model = onnx.load(onnx_model)

# 模型构建期
# 创建日志
# 可选参数 VERBOSE, INFO， WARNING, ERROR, INTERNAL_ERROR 产生不同等级的日志，由详细到睑裂
logger = trt.Logger(trt.Logger.ERROR)

# 引擎构建起，传入TensorRT引擎(builder)
builder = trt.Builder(logger)

# 使用TensorRT引擎(builder) 创建TensorRT网络(network)
# Explicit Batch 为TensorRT主流Network构建方法
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# 创建解析.onnx的解析器 并 检查解析过程中是否遇到任何错误的过程
# onnx模型的解析结果保存在 network中
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):  # onnx_model.SerializeToString()将ONNX模型对象序列化为字节串，这是TensorRT解析器期望的输入
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

# 网络属性选项
# ONNX模型被解析完成后，创建TenorRT引擎 配置器config,在此在引擎中配置了 最大工作空间、模型精度、优化配置文件（以便为 具有动态尺寸的输入提供优化）
config = builder.create_builder_config()
config.max_workspace_size = 1 << 20  # 位移运算表达式，1左移20位，1 * 2^20等于1,048,576字节，1024KB，即1MB
                                     # TensorRT引擎配置中的最大临时工作区大小设置为1MB
                                     # 表示 被移动数字 乘以2的幂（左移）或除以2的幂（右移）
 # 选择模型精度，在此使用FP16模式
  # 部分层可能精度下降导致较大误差
   # 找到误差较大的层（用polygraphy等工具）
   # 强制该层使用FP32进行计算
config.flags = 1 << int(trt.BuilderFlag.FP16)

# Dynamic Shape模式（除了Batch维度外，输入张量形状在推理时才决定网络）
 # 需要 Explict Batch 模式
 # 创建用于Dynamic Shape输入的配置器
profile = builder.create_optimization_profile()

 # 给定输入张量的 最小、最常见、最大尺寸尺寸
profile.set_shape('input', [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])  # 配置输入尺寸的优化轮廓
                                                                                  #'input'是输入张量的名称，接下来的三个尺寸数组分别代表最小尺寸、最优尺寸和最大尺寸
                                                                                  # 在这个特定的调用中，三个尺寸都设置为 [1, 3, 224, 224]，这意味着在这个场景中输入尺寸被认为是固定的
                                                                                  # 故实际上并未展示优化配置文件的动态尺寸处理能力

 # 将设置的profile传递给 config 以创建网络
config.add_optimization_profile(profile)

# 完成 network的结构 并 完成配置(config)后，传入network和config 创建cuda engine
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

# 输出.engine模型
with open('model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")
# 对生成的 TensorRT 模型 使之进行推理
from typing import Union, Optional, Sequence, Dict, Any
import torch
import tensorrt as trt

# "with ... as ..." 是一种用来简化 资源管理 和 异常处理 的语法结构，常用于文件操作、线程锁 需要手动关闭或释放资源的情况，
# 其核心作用是 确保无论代码块内的操作是否成功执行，相关的资源都会被正确且及时清理（比如 关闭文件、释放锁）
# with expression as variable:
  # 在此处的代码可以安全地使用'variable'
  # 无需担心资源管理的问题，如打开的文件会自动关闭
# e.g.
# with open('example.txt', 'r') as file:
#     content = file.read()
#     ...
# open('example.txt', 'r') 是表达式，它返回一个文件对象，对象在此命名为 'file'
# 当退出 with 块时，不管是 代码正常结束 或 异常，file对象都会被 自动关闭

# 运行期
class TRTWrapper(torch.nn.Module):
    # engine: Union[str, trt.ICudaEngine]: 表示'engine'参数可接受 str或trt.ICudaEngine
    # str: 一个字符串，通常是一个文件路径，指向已序列化的TensorRT CUDA引擎文件。
    # trt.ICudaEngine: 一个TensorRT的ICudaEngine对象，即'.engine'格式的模型

    # output_names: Optional[Sequence[str]] = None: 表示output_names参数是一个可选参数(Optional)，默认值为None
    #                                               它可以是一个序列（如列表、元组等），元素为字符串类型，用来指定模型输出的名称

    # -> None: 表示这个方法的返回类型是None，不要"-> None"默认返回None

    # __init__函数 用于 创建 self.engine、 self.context、并通过遍历 self.engine 创建 self._input_names、self._output_names
    def __init__(self, engine: Union[str, trt.ICudaEngine], output_names: Optional[Sequence[str]] = None)->None:
        super().__init__()
        self.engine = engine
        # 用于检查self.engine变量 是否为 字符串类型
        if isinstance(self.engine, str):
            # 使用上下文管理器创建 TensorRT的日志记录器 和 运行时环境
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                # 使用二进制模式('rb')打开之前 字符串传递的路径的文件
                with open(self.engine, mode='rb') as f:
                    # 读取整个文件内容到 engine_bytes中，这一步是将 引擎文件 加载到 内存中
                    engine_bytes = f.read()
                # 调用runtime的deserialize_cuda_engine方法，将序列化的引擎反序列化形成ICudaEngine对象
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        # 调用engine的create_execution_context()方法 创建用于执行推理的对象context（操作 发动机工作的"操作者",GPU进程）
        # context用于管理实际的模型执行，包括输入输出数据的绑定、执行推理等
        self.context = self.engine.create_execution_context()

        # 通过列表推导式遍历self.engine，将engine格式的模型的 输入和输出张量名称 收集到一个列表中
        names = [_ for _ in self.engine]

        # 使用filter函数 和 self.engine.binding_is_input方法，从所有绑定名称中筛选出代表输入（input）的名称
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    # inputs: Dict[str, torch.Tensor]: 这是方法的输入参数，它是一个类型注解，说明 inputs 应该是一个字典类型
    # 键（key）是字符串类型（str）
    # 值（value）是 torch.Tensor 类型，即PyTorch张量，用于存储模型的输入数据
    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        # bindings将存储模型 输入、输出张量的内存地址，初始化为[None, None, ... ,None]
        bindings = [None] * (len(self._input_names) + len(self._output_names))

        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            # profile是从TensorRT引擎获取特定profile的尺寸描述，是一个包含3个元素的元组(min_shape, opt_shape, max_shape)
            profile = self.engine.get_profile_shape(profile_id, input_name)
            # 检查输入张量的维度数，profile[0]是最小尺寸部分。如果不相等，程序会抛出错误，提示“Input dim is different from engine profile.”
            assert input_tensor.dim() == len(profile[0]), 'Input dim is different from engine profile.'
            # zip函数将这三个元组按元素配对
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            # 根据输入名称（input_name）查询并获取该输入在TensorRT引擎中的绑定索引（binding index）
            idx = self.engine.get_binding_index(input_name)

            # 确保input_tensor确实存储在GPU上
            assert 'cuda' in input_tensor.device.type
            # 确保内存连续性,在某些操作之后（如切片、转置等），张量的内存可能变得非连续，这可能会影响到将其数据复制到其他地方的效率，特别是当与底层库（如CUDA）交互时
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            # 调用set_binding_shape方法，可以让TensorRT引擎知道即将到来的推理批次的输入尺寸，这对于动态尺寸的模型是必要的
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # 为模型的所有输出在GPU上分配了内存，并为TensorRT执行准备好了输出绑定
        # 初始化一个空字典，用于存储模型输出张量，其中键为输出名称，值为对应的张量对象
        outputs = {}
        for output_name in self._output_names:
            # 对于每一个output_name，通过调用get_binding_index方法获取该输出在TensorRT引擎中的绑定索引
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            # 使用获得的索引idx，调用get_binding_shape方法从执行上下文中获取该输出的预期形状，并转换为Python元组
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch.device('cuda')  # 输出张量应当被创建在GPU上
            # 根据之前获取的 形状和数据类型，在指定的GPU设备上创建一个空的张量，这个张量将用于存放模型的输出结果。
            output = torch.empty(size=shape, dtype=dtype, device=device)
            # 创建的输出张量存储到字典outputs中，键为输出名称
            outputs[output_name] = output
            # 当前输出张量的内存地址 设置到 之前初始化的bindings列表中的相应位置
            bindings[idx] = output.data_ptr()

        # 异步地执行TensorRT上下文中的模型推理
        # bindings是包含 输入和输出张量的内存地址 指针的列表
        # torch.cuda.current_stream().cuda_stream 获取了 当前CUDA流的对象
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        return outputs

model = TRTWrapper('model.engine', ['output'])
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda()))  # 调用model的 forward函数
print(output)
# 现在抛开PyTorch，使用ONNX的Python API构造一个描述线性函数 output=a*x+b的ONNX模型
# ONNX模型按以下结构组织起来：

# · ModelProto
#  · GraphProto
#     · NodeProto
#     · ValueInfoProto

# 根据上面的结构，自底向上构造模型
# 使用 helper.make_tensor_value_info 构造出一个描述张量信息的 ValueInfoProto对象，其数据定义为 name:str  type:Enum  shape:list[int]
#                                                                                     (张量名称    数据类型       张量形状)

import onnx
from onnx import helper
from onnx import TensorProto

# 使用helper.make_tensor_value_info 构造出一个描述张量信息的 ValueInfoProto 对象，根据数据定义 张量名、张量的基本数据类型、 张量形状
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

# 构造算子节点信息NodeProto，可通过在helper.make_node中传入 算子类型、输入张量名、输出张量名。
# 先构造描述 c=a*x 的乘法节点，再构造 output=c+b 的加法节点
mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])

# 计算机中，图一般是用 一个节点集 和 一个边集 表示的，
# 而ONNX巧妙地把 边的信息保存在了节点信息里，省去了保存 边集的步骤
# ONNX中，如果某个节点的输入名 和 之前某个节点的输出名 相同，就默认这两个节点是相连的。 如上面的例子所示： 'Mul'节点定义输出张量c，
#                                                                                        'Add'节点定义输入张量c,
#                                                                                         则默认Mul节点 和 Add节点 是相连的。

# 判断ONNX模型 是否满足以上标准的接口: onnx.checker.check_model

# 使用helper.make_graph 构造计算图 GraphProto。
# helper.make_graph 函数需要传入 节点、图名称、输入张量信息、输出张量信息 4个参数
# 节点参数 必须以 拓扑序 给出（即 计算图自上而下的算子运算顺序 与 节点参数 要吻合），拓扑序 是与 有向图 相关的数学概念
graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])

# 最后使用helper.make_model 把 计算图 GraphProto 封装进 ModelProto 里，一个ONNX模型就构造完成了
model = helper.make_model(graph)

# 检查模型正确性、把模型以文本形式输出、存储到一个".onnx"文件里，
# 使用 onnx.checker.check_model检查模型是否满足ONNX标准是必要的，因为无论模型是否满足标准，ONNX都允许我们用onnx.save存储模型
onnx.checker.check_model(model)
print(model)
onnx.save(model, 'linear_func.onnx')

# 用ONNX Runtime运行模型，验证模型是否正确:
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('linear_func.onnx')

a = np.random.rand(10, 10).astype(np.float32)
b = np.random.rand(10, 10).astype(np.float32)
x = np.random.rand(10, 10).astype(np.float32)

output = sess.run(['output'], {'a': a, 'b': b, 'x': x})[0]

assert np.allclose(output, a * x + b)

# 总结：
# 1. ONNX 使用 Protobuf 定义规范和序列化 模型
# 2. 一个ONNX模型由 ModelProto, GraphProto, NodeProto, ValueInfoProto 这几个数据类的对象组成
#    构建ONNX模型时，顺序是Tensor->Node->Graph->Model
#    tensor = onnx.helper.make_tensor_value_info(...)
#    node = onnx.helper.make_node(...)
#    graph = onnx.helper.make_graph(...)
#    model = onnx.helper.make_model(graph)


# case3: 一种简单的为PyTorch添加C++算子实现的方法，来代替较为复杂的新增TorchScript算子

# 使用Python的setuptools编译功能 和 PyTorch的C++拓展工具函数，能够编译包含了torch库的C++源文件
# 编写如下的 Python 代码，编译my_add.cpp文件 并 导出为 动态库.so(Shared Object)
from setuptools import setup
from torch.utils import cpp_extension

setup(name='my_add',
      ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

# 在终端执行 python3 03_setup.py develop 即可生成my_lib
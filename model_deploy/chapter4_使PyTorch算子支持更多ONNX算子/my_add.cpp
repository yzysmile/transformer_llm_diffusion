// case3: 一种简单的为PyTorch添加C++算子实现的方法，来代替较为复杂的新增TorchScript算子

#include <torch/torch.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b)
{
    // torch::Tensor 就是 C++ 中 torch 的张量类型，它的加法和乘法等运算符均已重载
    return 2 * a + b;
}

// 为C++函数提供Python接口，my_lib 是未来在 Python中 导入的模块名
 // 第一个参数： 双引号中的"my_add"是 Python调用接口的名称
 // 第二个参数： my_add是 C++函数名字
PYBIND11_MODULE(my_lib, m)
{
    m.def("my_add", my_add);
}
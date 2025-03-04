import math
import copy  # 提供 浅拷贝 和 深拷贝
from pathlib import Path  # 方便进行路径操作
from random import random  # 随机数生成函数
from functools import partial  # 可用于对函数（类）进行部分参数绑定（固定）
from collections import namedtuple  # 提供类似结构体的命名元组
from multiprocessing import  cpu_count  # 获取当前机器的CPU核心数

import torch
from numpy.ma.core import remainder
from pyparsing import alphas
from sympy.geometry.entity import scale
from torch import nn, einsum  # einsum 用于爱因斯坦求和
import torch.nn.functional as F  # 提供常用函数式神经网络操作，如卷积、池化等
from torch.utils.data import Dataset, DataLoader  # 数据集基类和数据加载器

from torch.optim import Adam

import torchvision  # PyTorch视觉工具包
from torchvision import transforms as T, utils  # 图像预处理transforms, utils 提供图像显示保存等功能

from einops import rearrange, reduce, repeat  # einops提供灵活的张量变换函数
# einops 是一个用于张量操作的 Python 库，如重排（rearrange）、缩减（reduce）和重复（repeat）

from einops.layers.torch import Rearrange  # einops 在 Pytorch中的Layer实现

from PIL import Image  # Python Image Library，图像读写操作
from tqdm.auto import tqdm  # 进度条库
from ema_pytorch import EMA  # 指数滑动平均库，用于模型权重的EMA

from accelerate import Accelerator  # 用于分布式训练加速库
import matplotlib.pyplot as plt  # 常用的绘图库
import os  # 与操作系统相关的功能

torch.backends.cudnn.benchmark = True

torch.manual_seed(4096)  # 设置随机数种子，确保生成的张量相同

if torch.cuda.is_available():
    torch.cuda.manual_seed(4096)

def linear_beta_schedule(timesteps):
    """
    定义beta参数 在每个时间步的取值
    """
    scale = 1000/timesteps  # 用timesteps对原始范围进行缩放
    beta_start = scale * 0.0001  # 线性Beta的开始值
    beta_end = scale * 0.02  # 线性Beta的结束值
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    # 在[beta_start, beta_end]范围上， 生成timesteps个等差数列。

def extract(a, t, x_shape):
    """
    从向量a中取出时间步t对应的值，并reshape成x_shape
    """
    b, *_ = t.shape  # 获取batch大小，并忽略batch之后的所有维度
    out = a.gather(-1, t)  # 在a的最后一个维度上根据 t 提取对应的值
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    # reshape 成 (b, 1, 1, ... ,1) 的形式

class Dataset(Dataset):
    """
    自定义数据集，用于加载指定文件夹下的.jpg图像文件
    """
    def __init__(self,
                 folder,
                 image_size
    ):
        self.folder = folder  # 图像文件所在的文件夹路径
        self.image_size = image_size  # 保存图像尺寸
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]
        # .glob(f'**/*.jpg') 递归搜索该目录下所有子目录中的.jpg文件
        # 在pathlib中，**直接触发递归，无需额外配置。

        self.transform = T.Compose([
            T.Resize(image_size),  # 调整图像大小为 image_size * image_size
            T.ToTensor()  # 将图像转换为Pytorch张量，并将像素归一化到[0, 1]
        ])

    def __len__(self):
        # 返回数据集的大小
        return len(self.paths)

    def __getitem__(self, index):
        # 根据索引返回对应的图像数据
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def exists(x):
    return x is not None  # 如果x不是None, 则返回True, 否则返回False

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
    # 检查给定的参数 val 是否存在（即是否为有效值），如果 val 存在，则返回 val；否则，根据 d 的类型返回相应的默认值
    # print(default(5, 10))  # 输出: 5
    # print(default(None, 10))  # 输出: 10
    # print(default(None, lambda: 10 + 5))  # 输出: 15

def identity(t, *args, **kwargs):
    return t
    # 一个简单的恒等函数，原样返回输入t，本质不对数据做任何处理

def cycle(dl):  # dl即dataloader
    while True:
        for data in dl:
            yield data
    # 生成器，循环迭代给定的dataloader(dl)，使其可无限迭代

def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num
    # 判断num的平方根是否为整数
    # math.sqrt(num)**2 若等于num，则num的平方根是整数

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    # 将num拆分成大小为 divisor 的若干组，如果最后有余数 remainder，则将它作为一组附加到数组的末尾
    # num = 10, divisor=3, 返回[3, 3, 4]

def normalize_to_neg_one_to_one(img):
    return img*2 - 1
    # 将图像像素值[0, 1]的范围映射到[-1, 1]

def unnormalize_to_zero_to_one(t):
    return (t+1) * 0.5
    # 将图像像素值[-1, 1]的范围映射回[0, 1]

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
        # 前向传播时，将输入x经过self.fn，再加回原始x实现残差结构

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )
    # 上采样模块：
    # 1) 将特征图放大2倍，采用最邻近插值
    # 2) 卷积将通道数从 dim 映射到 dim_out(如果dim_out没传，则仍为dim)

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
    # 下采样模块：
    # 1) 通过 einops 将高宽各自分辨率乘以2的模式展平到通道维度上
    #    使得通道数扩大4倍(p1 = 2, p2 = 2 => 2*2=4)
    # 2）1*1 卷积 将通道数映射到 dim_out

class WeightStandardizedConv2d(nn.Conv2d):
    """
    weight standardization purportedly works synergistically with group normalization
    权重标准化（Weight Standardization）据称与组归一化（Group Normalization）具有协同作用，
    同时使用权重标准化和组归一化这两种技术可以带来更好的效果
    """

    # 权重标准化:一种针对卷积层权重的归一化方法，旨在使每一层的权重具有零均值和单位方差。
    # 将输入特征分成若干组，并分别对每组内的特征进行归一化
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # 根据数据类型选择不同的数值稳定性常数 eps

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        # 'o ... -> o 1 1 1'：表示将输入张量的第一个维度保留，其余维度保留均值
        # 'mean'：表示在缩减过程中计算均值。

        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbias = False))
        # 计算卷积核再输出通道维度o上的 均值mean 和 方差var
        # partial(torch.var, unbiased=False)：使用 torch.var 函数计算方差，并设置 unbiased=False 以确保计算的是有偏估计。

        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        # 对卷积核进行标准化

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        # 得到标准化后的卷积操作

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        # 可学习的缩放参数g， 初始化为1

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g
        # 沿着通道维度(=1)做 layer norm，并使用g来进行缩放

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
        # 在执行fn前，先进行LayerNorm

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
        # 前向过程： 先归一化，再执行传入的fn

# sinusoidal positional embeds
# 一个给定的位置pos 和 维度i
# i，正余弦嵌入的公式如下：
# PE(pos, 2i) = sin( pos/10000^(2i/d) )
# PE(pos, 2i+1) = cos( pos/10000^(2i/d) )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 记录嵌入维度

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2  #  嵌入维度的一半，每个位置嵌入由正弦和余弦两部分组成
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 生成等比数列，以用于构建正余弦频率

        emb = x[:,None] * emb[None, :]
        # 将输入张量 x 的形状从 (batch_size,) 转换为 (batch_size, 1)
        # 将输入时间步x与频率emb相乘

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    可学习的 sinusoidal pos emb
    """
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)
        # 如果 is_random 为 False，则这些频率是可学习的；否则是随机固定

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        # 将输入 x reshape 成 (batch, 1)

        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 计算随机或可学习的频率 freqs

        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # 拼接正弦和余弦

        fouriered = torch.cat((x, fouriered), dim=-1)
        # 再将原始 x 与正余弦部分合并

        return fouriered
        # 返回包含输入和正余弦编码的结果


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        # self.proj——标准化后的卷积操作

        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        # 标准化后的卷积 -> GroupNorm -> SiLU激活

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
            # 若从时间嵌入中得到 scale_shift，则对特征图进行缩放和偏移

        x = self.act(x)
        return x  # 输出经过标准化和激活的张量


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        # 如果传入了 time_emb_dim，则对时间嵌入进行线性映射得到 scale 和 shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # 如果 dim != dim_out，就用 1x1 卷积在残差分支中对通道数进行调整

    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            # 将 time_emb 拆分为 (scale, shift)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)  # 最终输出为正常流 (h) + 残差分支


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 一次性生成 q, k, v
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )
        # 输出层（卷积 + LayerNorm）

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将通道维分成 q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        # 分别对 q 的通道维(-2)和 k 的序列维(-1)做 softmax

        q = q * self.scale
        v = v / (h * w)
        # 缩放 q，以及对 v 做归一化

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        # 先将 k 和 v 做乘积，得到上下文 context

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # 再和 q 做乘积以得到输出

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        # reshape 回原始形状

        return self.to_out(out)
        # 卷积 + LayerNorm 得到最终结果

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        # 自注意力机制：先获取 q, k, v，再做注意力加权求和，最后映射回 dim

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale
        # 缩放 q

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        # 相似度矩阵 sim (b, heads, i, j)

        attn = sim.softmax(dim = -1)
        # 沿着最后一维做 softmax，得到注意力分布

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        # 加权求和得到输出

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        # reshape 回原始分辨率

        return self.to_out(out)  # 最后再用 1x1 卷积映射回 dim 维度

# model 需要减去的噪声是由U-Net进行预测的
# 因为模型每一次去噪时，使用的都是同一个模型，故要Timestep来告诉模型现在现在进行的具体是哪一步去噪
class Unet(nn.Module):
    def __init__(self,
                 dim,
                 init_dim = None,
                 out_dim = None,
                 dim_mults = (1, 2, 4, 8),  #  表示每一层相对于初始维度 dim 的倍数变化
                 channels = 3,
                 resnet_block_groups = 8,
                 learned_sinusoidal_cond = False,  # 是否使用学习到的正弦条件
                 random_fourier_features = False,  # 是否使用随机傅里叶特征
                 learned_sinusoidal_dim = 16
    ):
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)
        # 输入通道 -> init_dim，使用7*7卷积做初始特征提取

        dims = [init_dim, * map(lambda m: dim * m, dim_mults)]
        # lambda arguments: expression；lambda m: dim * m 是一个匿名函数，它接受一个参数 m，并返回 dim * m 的结果

        # map将一个函数应用到 列表、元组的每一个元素上
        # map(lambda m: dim * m, dim_mults)：使用map函数和lambda表达式将dim_mults中的每个元素乘以dim，生成新的列表
        # “*” 将 map 函数的结果解包成单独的元素，并与 init_dim 组合成一个新的列表。

        in_out = list(zip(dims[:-1], dims[1:]))
        # dim = 64时， dim_mults=(1, 2, 4, 8)，则dims=[64, 64*1, 64*2, 64*4, 64*8]
        # zip将2个list中的元素进行配对
        # in_out 就是 [(64, 64), (64, 128), (128, 256), (256, 512)]

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        # partial 用来部分应用一个类的构造函数 ResnetBlock，并固定其 groups 参数
        # block_klass 是通过部分应用 ResnetBlock类 生成的新类构造函数，它在创建对象时会自动将groups参数默认设置为8

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        # 根据需要选择使用随机/可学习的正弦嵌入，或使用经典的正弦嵌入

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # 时间嵌入先进入正弦嵌入，然后用两个全连接层(中间激活为 GELU)，维度转为time_dim

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)  # 判断是否是最后一个分辨率

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            # down 阶段：
            # 1) ResnetBlock(dim_in -> dim_in)
            # 2) 再一个ResnetBlock(dim_in -> dim_in)
            # 3) Residual(PreNorm(LinearAttention))
            # 4) 如果不是最后层，用Downsample；否则用3*3卷积保持分别率

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        # 中间层(U-Net 最底部)： ResnetBlock -> 自注意力 -> ResnetBlock

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            # 倒序遍历 in_out，用于 up 阶段

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            # up 阶段的逻辑与 down 类似，只是要先拼接 skip connection

        self.out_dim = default(out_dim, channels)
        # 最终输出通道数，默认与输入通道一致

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        # 最后一步和初始输入拼接后，再过一个 ResnetBlock

        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        # 通过 1x1 卷积将维度映射到 out_dim

    def forward(self, x, time):
        x = self.init_conv(x)  #  初始卷积提取特征
        r = x.clone()  # 保存初始特征用于最后拼接

        t = self.time_mlp(time)  # 将时间步 time 通过 time_mlp 得到时间嵌入t

        h = []  # 用于保存每层的输出，以便再解码器阶段做 skip connection

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            # 依次执行 block1 -> block2 -> attn -> downsample
            # 并存储中间输出h

        # mid
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            # pop出 下采样时存储的输出，进行skip connection
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        # final
        x = torch.cat((x, r), dim = 1)  # 与最初的输入特征r拼接

        x = self.final_res_block(x, t)
        return self.final_conv(x)  # 最终输出一个与输入维度相匹配的特征图

model = Unet(64)  # 实例化一个U-Net模型，基本通道数dim=64

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,  # 传入的U-Net模型，用于预测噪声
            *,
            image_size,
            timesteps = 1000,  # 扩散过程的总时间步数
            beta_schedule = 'linear',
            auto_normalize = True  # 是否自动将图像[0, 1]归一化到[-1,1]
    ):
        super().__init__()  # 继承自 nn.Module

        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # 如果是 GaussianDiffusion类 本身，则要求model的输入通道 和 输出通道一致，否则报错

        assert not model.random_or_learned_sinusoidal_cond
        # 不允许网络使用随即或可学习的正弦位置编码

        self.model = model
        self.channels = self.model.channels  # 图像的通道，默认为3

        self.image_size = image_size  # 保存图像大小

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        else:
            raise  ValueError(f'unknown beta schedule {beta_schedule}')
        # 根据传入的 beta_schedule 字符串选择 beta 调度函数，目前只支持'linear'，否则抛出异常

        betas = beta_schedule_fn(timesteps)  # 计算在每个时间步上的beta值（线性递增）

        alphas = 1. - betas
        # α_t = 1 - β_t

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # 累乘得到 α_1 * α_2 * ... * α_t

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
        # 向前偏移一个时间步，便于在计算 q(x_{t-1}|x_t, x_0) 时使用
        # 第一个时间步补 1，使 α_cumprod_prev 的长度与 alphas_cumprod 一致

        timesteps, = betas.shape
        # 获取时间步数（1000）
        self.num_timesteps = int(timesteps)
        # 将其保存为整型

        # sampling related parameters
        self.sampling_timesteps = timesteps
        # 采样时使用的步数，默认和训练步数相同

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        # 定义一个小函数，用于将各种张量注册为 buffer，并转换为 float32 类型

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # 将以上计算好的 beta、alpha 累乘、以及前一个时间步的 alpha 累乘注册为 buffer
        # 这些值是训练和推理都会用到，但不会被训练的参数

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # sqrt(累乘α_t)
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # sqrt(1 - 累乘α_t)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # 记录 log(1 - 累乘α_t)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # sqrt(1 / 累乘α_t)
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # sqrt(1 / 累乘α_t - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # q(x_{t-1} | x_t, x_0) 的后验方差
        # 根据公式： posterior_variance_t = β_t * (1 - α_{t-1}累乘) / (1 - α_t累乘)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)
        # 注册后验方差

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # 取对数时夹紧最小值防止数值溢出
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # 后验均值系数1
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # 后验均值系数2

        # derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        # SNR = α_t累乘 / (1 - α_t累乘)

        # https://arxiv.org/abs/2303.09556
        maybe_clipped_snr = snr.clone()
        # 这里可以对 snr 做一些裁剪操作，如果需要的话

        register_buffer('loss_weight', maybe_clipped_snr / snr)
        # 用于加权损失的系数

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        # 根据 auto_normalize 决定是否对数据进行 [-1,1] <-> [0,1] 的转换

    def predict_start_from_noise(self, x_t, t, noise):
        """
        通过 x_t 和噪声，反推 x_0 的预测值
        x_0 = 1 / sqrt(alpha_cumprod) * x_t - sqrt(1 / alpha_cumprod - 1) * noise
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        通过 x_t 和对 x_0 的预测值，反推噪声的预测值
        noise = (1 / sqrt(alpha_cumprod) * x_t - x_0) / sqrt(1 / alpha_cumprod - 1)
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 后验分布的均值
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # 后验分布的方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        # 后验分布方差的对数（已做 clip）
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        """
        给定当前噪声图 x 和时间步 t，通过模型预测噪声 pred_noise，并得到对 x_0 的估计 x_start
        """
        model_output = self.model(x, t)
        # 模型输出，通常是预测噪声

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        # 如果需要对预测出的 x_0 做裁剪，则 partial(torch.clamp)；否则恒等函数

        pred_noise = model_output
        # 这里把模型输出视为噪声预测
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        # 对 x_0 进行 [-1,1] 裁剪（可选）

        if clip_x_start and rederive_pred_noise:
            # 如果 x_0 被裁剪，为了更准确，需要重新计算一次噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        """
        计算从扩散过程中 p(x_{t-1} | x_t) 的均值和方差，用于反向采样
        """
        noise, x_start = self.model_predictions(x, t)
        # 模型预测噪声和 x_0

        if clip_denoised:
            x_start.clamp_(-1., 1.)
            # 默认会把 x_0 的范围裁剪到 [-1,1]

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start,
            x_t=x,
            t=t
        )
        # 计算后验分布的均值和方差
        # 这里的后验分布相当于 q(x_{t-1}|x_t, x_0)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int):
        """
        在反向扩散的某一个时间步 t，从 p(x_{t-1} | x_t) 采样
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        # 构造与批大小相同的时间张量

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            clip_denoised=True
        )
        # 根据 x_t 计算后验均值和方差

        noise = torch.randn_like(x) if t > 0 else 0.
        # 如果 t > 0 则在采样时加噪声；如果 t=0，则不再加噪声

        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # 采样公式： x_{t-1} = 均值 + 标准差 * 噪声

        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        """
        从纯噪声开始，逐步反向采样还原图像
        """
        batch, device = shape[0], self.betas.device
        # batch 大小, 使用存储在 buffer 中的 betas 的设备

        img = torch.randn(shape, device=device)
        # 初始从标准正态分布采样

        imgs = [img]
        # 用于保存采样过程中每个时间步的结果
        x_start = None

        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # 从 T-1 到 0 逐步反向采样
            img, x_start = self.p_sample(img, t)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        # 如果 return_all_timesteps=True, 返回整个采样序列；否则只返回最终生成的图像

        ret = self.unnormalize(ret)
        # 将图像从 [-1,1] 转回 [0,1]
        return ret

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        """
        对外提供的采样接口
        """
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
        # 默认使用 p_sample_loop 进行逐步采样
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：从 x_0 得到 x_t 的采样
        x_t = sqrt(α_cumprod) * x_0 + sqrt(1-α_cumprod) * noise
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 如果不指定噪声，则生成一个和 x_start 形状相同的高斯噪声

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        return F.mse_loss
        # 训练时使用的损失函数，默认是 MSE

    def p_losses(self, x_start, t, noise=None):
        """
        在给定 x_0 以及随机的时间步 t 时，计算训练时的损失
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 前向扩散，将 x_0 添加噪声到 x_t

        # predict and take gradient step
        model_out = self.model(x, t)
        # 模型对 x_t 进行估计噪声

        loss = self.loss_fn(model_out, noise, reduction='none')
        # 计算 MSE 损失 (逐元素)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # 在除 batch 之外的所有维度取平均 (即每个样本的损失)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        # 乘以权重 (与 SNR 相关)

        return loss.mean()
        # 返回对整个 batch 的平均损失

    def forward(self, img, *args, **kwargs):
        """
        模块的前向调用接口，一般在训练时调用
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 解包图像形状、设备以及定义的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # 随机采样一个时间步 t 用于训练

        img = self.normalize(img)
        # 如果开启了 auto_normalize，则把 [0,1] 的图片映射到 [-1,1]

        return self.p_losses(img, t, *args, **kwargs)
        # 调用 p_losses 计算训练损失

class Trainer(object):
    def __init__(
        self,
        diffusion_model,                  # 传入的扩散模型 (GaussianDiffusion)，内部封装了 U-Net 等
        folder,                           # 数据存放的文件夹路径
        *,
        train_batch_size = 16,           # 训练批大小
        gradient_accumulate_every = 1,   # 梯度累积步数
        train_lr = 1e-4,                 # 学习率
        train_num_steps = 100000,        # 总训练步数
        ema_update_every = 10,           # 每多少步进行一次 EMA 更新
        ema_decay = 0.995,               # EMA 的衰减率
        adam_betas = (0.9, 0.99),        # Adam 优化器的 betas 参数
        save_and_sample_every = 1000,    # 每隔多少步进行一次模型保存及采样
        num_samples = 25,                # 采样时生成的图像数量
        results_folder = './results',    # 存放训练结果（模型、采样图像）的文件夹
        split_batches = True,            # 是否在多卡或大批次下自动拆分 batch
        inception_block_idx = 2048       # 不常用，此处保留的参数，一般与 FID 计算相关
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'no'
        )
        # 使用 accelerate 库加速训练，split_batches 控制是否拆分大 batch，
        # mixed_precision 设为 'no' 表示不使用混合精度

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        # 保存传入的扩散模型和其通道数

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        # 检查 num_samples 是否为完全平方数以便后续可视化；
        # 设置本类所需的各种超参数：批大小、梯度累积次数、总训练步数、图像大小等

        # dataset and dataloader
        self.ds = Dataset(folder, self.image_size)
        # 使用自定义 Dataset 读取文件夹下的数据，并将图像缩放到指定大小
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        # 创建 PyTorch DataLoader，用于批量加载数据
        dl = self.accelerator.prepare(dl)
        # 将 DataLoader 传给 accelerate，处理多卡或分布式训练场景
        self.dl = cycle(dl)
        # 使用 cycle 函数，使得 dataloader 可以被无限迭代

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        # 使用 Adam 优化器，对扩散模型的所有参数进行优化

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        # 只有主进程才创建 EMA 对象；EMA 内部会复制一份模型用于平滑更新

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        # 创建存放结果的文件夹（若不存在则自动创建）

        # step counter state
        self.step = 0
        # 训练步计数器

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        # 使用 accelerate 准备模型和优化器（比如放到 GPU 或多卡同步等）

    @property
    def device(self):
        return self.accelerator.device
        # 方便获取当前加速器所使用的设备（GPU/CPU）

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        # 若不是本地主进程，则不做模型保存

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        # 打包当前训练状态：步数、模型权重、优化器状态、EMA 状态，以及混合精度 scaler 状态

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        # 保存到指定路径

    def load(self, ckpt):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)
        # 加载检查点文件到当前设备

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        # 从 checkpoints 中恢复模型权重

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # 恢复当前训练步数和优化器状态

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
        # 恢复 EMA 权重

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        # 若使用了混合精度，则恢复 scaler 状态

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            # 使用 tqdm 显示进度条，仅在主进程打印

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    # 从循环的 dataloader 中获取一批图像，并移动到当前训练设备

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        # 前向传播，得到损失
                        loss = loss / self.gradient_accumulate_every
                        # 根据梯度累积步数进行损失缩放
                        total_loss += loss.item()
                        # 累加损失值，方便输出到进度条

                    self.accelerator.backward(loss)
                    # 反向传播，累加梯度（accelerate 可能在多卡情景下做同步）

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # 对梯度进行裁剪，防止梯度爆炸
                pbar.set_description(f'loss: {total_loss:.4f}')
                # 更新进度条的描述

                accelerator.wait_for_everyone()
                # 在多卡训练中等待所有卡的梯度都准备好再更新

                self.opt.step()
                self.opt.zero_grad()
                # 优化器更新参数，并清空梯度

                accelerator.wait_for_everyone()

                self.step += 1
                # 训练步 +1

                if accelerator.is_main_process:
                    # 只有主进程才执行以下保存/采样操作
                    self.ema.update()
                    # EMA 更新

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 每隔 save_and_sample_every 步执行一次模型保存和图像采样
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # 将要生成的 num_samples 张图像分成若干批
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            # 分批调用 sample 函数生成图像

                        all_images = torch.cat(all_images_list, dim = 0)
                        # 把分批生成的图像拼接

                        utils.save_image(
                            all_images,
                            str(self.results_folder / f'sample-{milestone}.png'),
                            nrow = int(math.sqrt(self.num_samples))
                        )
                        # 保存生成图像到本地文件夹，并用 nrow 控制每行图像数量

                        self.save(milestone)
                        # 保存当前模型和训练状态

                pbar.update(1)
                # 进度条 +1

        accelerator.print('training complete')
        # 在所有进程上打印训练完成提示（accelerator.print 会在多卡训练时同步）

    def inference(self, num=1000, n_iter=5, output_path='./submission'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # 若指定输出路径不存在则创建

        with torch.no_grad():
            for i in range(n_iter):
                batches = num_to_groups(num // n_iter, 200)
                # 将要生成的数量按批次划分
                all_images = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))[0]
                # 调用 EMA 模型生成图像，只取返回列表中的第一个 batch（此处写法较固定）

                for j in range(all_images.size(0)):
                    # 逐张保存
                    torchvision.utils.save_image(
                        all_images[j],
                        f'{output_path}/{i * 200 + j + 1}.jpg'
                    )
        # 该函数主要用于在训练完成后批量生成并保存图像

path = './faces/faces'
# 数据所在的文件路径，这里假设所有训练图像都在 ./faces/faces 目录中

IMG_SIZE = 64
# 设置图像尺寸为 64x64

batch_size = 16
# 设置训练时的批大小为 16 张图像

train_num_steps = 10000
# 训练的总步数，指优化器更新（iteration）次数

lr = 1e-3
# 学习率 (learning rate)，这里设置为 0.001

grad_steps = 1
# 梯度累积步数；若设置大于 1 则表示每累积一定次数的反向传播再进行一次优化更新

ema_decay = 0.995
# 指数移动平均 (EMA) 的衰减率，常用于在训练过程中平滑模型权重

channels = 16
# U-Net 的基础通道数，即第一个卷积层的通道数

dim_mults = (1, 2, 4)
# 用来指定 U-Net 不同下采样 / 上采样阶段的通道扩展倍数，
# 最终网络结构中的通道数将按 (channels, 2*channels, 4*channels, ...) 的形式逐步增加

timesteps = 100
# 扩散过程中加噪声的时间步数 T；比如在 DDPM 中可以是 1000，这里设置为 100

beta_schedule = 'linear'
# beta 的调度方式（表示在扩散过程中 beta 的变化），此处设置为线性

model = Unet(
    dim = channels,
    dim_mults = dim_mults
)
# 实例化一个 U-Net 模型对象，输入的基本通道数为 16，
# 会根据 dim_mults 逐步在网络层中增加通道数

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)
# 将 U-Net 模型封装到 GaussianDiffusion 类中，
# 并设置扩散过程中的一些参数（如图像大小、时间步数等）。
# 该类会负责前向扩散（加噪）和反向扩散（去噪）的具体实现。

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 1000
)
# 实例化一个 Trainer 类来管理训练流程：
# - 使用 diffusion 模型进行前向与反向传播
# - 每个 batch 的大小为 16
# - 使用学习率 1e-3
# - 总训练步数为 10000
# - 每个 step 都更新梯度（grad_steps=1）
# - EMA 衰减因子为 0.995
# - 每 1000 步保存一次模型并进行一次采样

trainer.train()
# 开始训练，Trainer 内部会执行循环读取数据、前向计算、损失反传、优化器更新等流程。

















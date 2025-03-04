import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
context_length = 128  # 128个token
d_model = 512  # 每一个token的 特征维度为128维
num_blocks = 12  # transformer blocks数量
num_heads = 8  # 8头注意力机制
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(  # ffn-feedforward network
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# 单头注意力机制
class Attention(nn.Module):
    def __init__(self, head_size: int):  # 类的构造函数接受一个名为'head_size'的参数，类型为int
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size  # 单头维度, e.g head_size为64
        self.context_length = context_length
        self.dropout = dropout  # 0.1

        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(  # 缓冲区名称是"tril"
            torch.ones((self.context_length, self.context_length))))  # 下三角为"1"，上三角位置为"0"的mask(tril是"triangular lower")
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps(current context_length), d_model(Channels dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        #  Q @ K^T / sqrt(d_k)  K.transpose(-2, -1)的作用是将 最后一个维度 和 倒数第二个维度 进行转置
        #  这里无法使用K.permute(-2, -1),K是一个三维张量K.permute的使用一定是K.permute(x, x, x)
        weights = (Q @ K.transpose(-2, -1)) * (
                    1.0 / math.sqrt(K.size(-1)))  # K.size(-1)返回的是K张量的最后一个维度的大小 e.g [4, 16, 64]中的64
        # Apply mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 将为"0"位置（上三角）替换为"-inf"
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout_layer(weights)

        #  Apply dot product attention: weights @ V
        out = weights @ V
        return out


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads  # 头数
        self.head_size = head_size  # 单头的维度
        self.d_model = d_model  # 总维度
        self.context_length = context_length  # 上下文长度
        self.dropout = dropout

        #  nn.Sequential 用于定义顺序组合的网络，而 nn.ModuleList 则用于保存一组子模块
        #  '_' 作为循环变量通常表示一个占位符，用于表示不需要使用的循环变量
        # 创建了一个包括5个Attention的nn.ModuleList
        self.heads = nn.ModuleList(Attention(head_size=self.head_size) for _ in range(self.num_heads))
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # 在循环中得"h模块"，并将x作为输入传递给"h模块"
        out = [h(x) for h in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feedforward_network = FeedforwardNetwork()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        x = x + self.multi_head_attention_layer(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, max_token_value=100080):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1,
                                                         embedding_dim=self.d_model)  # 随机初始化，训练过程中将不断优化
        # '*'表示对列表[]进行解压，将列表中的每个模块作为单独的参数传递给 nn.Sequential
        #  这样的写法等效于将每个模块分别传递给 nn.Sequential，并将它们按照给定的顺序组合成一个顺序容器
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.linear_after_transformer_blocks = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):  # 推理时只需要idx(批次*context_length)； 训练时需要idx和target，targets是idx的后错一位
        B, T = idx.shape  # T表示time
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model, device=device)
        position = torch.arange(0, self.context_length, dtype=float).unsqueeze(
            1)  # unsqueeze(1)表示添加第二个维度：e.g. 形状为(3,)的一维张量[0, 1, 2],
        # 对该张量进行unsqueeze操作，指定在维度1上添加维度，结果是一个形状为(3, 1)的二维张量，即tensor([[0.],
        #                                                                                  [1.],
        #                                                                                  [2.])
        div_term = (
            torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))).unsqueeze(
            0)  # 范围为【0， d_model），间隔为2， div_term是一个d_model/2维的一维张量
        position_encoding_lookup_table[:, 0::2] = torch.sin(
            position * div_term)  # torch.sin(position * div_term)是一个(16, 32)的二维张量
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding  # x维度[4, 16, 64]
        x = self.transformer_blocks(x)  # e.g x维度为[4, 16, 64]

        logits = self.linear_after_transformer_blocks(
            x)  # logits维度为[4, 16, max_token_value]，对上下文中的每一个字都要作预测？ 还是 对上下文的下一个字作预测？
        #  F.softmax(logits, dim=-1)  # 在max_token_value维度上作softmax

        if targets is not None:  # 若是训练过程
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)  # 使得每一行代表一个预测的概率分布
            targets_reshaped = targets.view(B * T)  # 使得每个元素表示一个样本的目标类别
            loss = F.cross_entropy(input=logits_reshaped,
                                   target=targets_reshaped)  # F.cross_entropy函数会在内部对输入的概率分布进行softmax操作
        else:
            loss = None  # 验证过程
        return logits, loss

    # 生成预测的文字
    def generate(self, idx, max_new_tokens=100):
        # idx 是 （B， T）的二维张量
        for _ in range(max_new_tokens):
            # 裁剪： 取所有行 并 从每行的倒数第 self.context_length 个位置开始一直取到末尾
            idx_crop = idx[:, -context_length:]
            # 获取预测值
            logits, loss = self.forward(idx_crop)  # logits是[batch_size, context_length, max_token_value]
            logits_last_timestep = logits[:, -1, :]  # logits_last_timestep维度为[4, max_token_value]， 表示上下文的下一个预测字 的概率分布

            probs = F.softmax(input=logits_last_timestep, dim=-1)

            idx_next = torch.multinomial(input=probs, num_samples=1)  # 根据概率分布进行采样，得到每个上下文 下一个最有可能的文字

            idx = torch.cat((idx, idx_next), dim=1)  # 上下文与预测值进行拼接
        return idx

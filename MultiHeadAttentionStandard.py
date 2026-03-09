import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力计算。
    
    参数:
        Q: 查询矩阵 (batch_size, seq_len_q, head_dim)
        K: 键矩阵 (batch_size, seq_len_k, head_dim)
        V: 值矩阵 (batch_size, seq_len_v, head_dim)
        mask: 通常(batch_size, seq_len_q, seq_len_k) 可选

    返回:
        output: (batch_size, seq_len_q, head_dim)
        attention_weights: (batch_size, seq_len_q, seq_len_k)
    """
    head_dim = Q.size(-1)  # head_dim
    
    # 计算点积并进行缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和，计算输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
    
 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        """
            embed_size: 输入序列的嵌入维度。
            每个 head 的维度:dim_k=embed_size/num_heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        # 为每个头单独定义 Q, K, V 的线性层，输出维度同为 head_dim
        self.w_q = nn.ModuleList([nn.Linear(embed_size, self.head_dim ) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(embed_size, self.head_dim ) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(embed_size, self.head_dim ) for _ in range(num_heads)])

        # 输出线性层，用于将多头拼接后的输出映射回 embed_size
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。
        
        参数:
            q: 查询矩阵 (batch_size, seq_len_q, embed_size)
            k: 键矩阵 (batch_size, seq_len_k, embed_size)
            v: 值矩阵 (batch_size, seq_len_v, embed_size)
            mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_k)

        返回:
            out: 注意力加权后的输出
        """
        multi_head_outputs = []
        # 对每个头分别计算 Q, K, V，并执行缩放点积注意力
        for i in range(self.num_heads):
            Q = self.w_q[i](q)  # (batch_size, seq_len_q, head_dim)
            # 用第 i 个 attention head 的 Query 投影矩阵，对输入 q 做一次线性变换，得到该 head 的 Q。
            # self.w_q[i]第 i 个 head 专属的一个 nn.Linear 层
            K = self.w_k[i](k)  # (batch_size, seq_len_q, head_dim)
            V = self.w_v[i](v)  # (batch_size, seq_len_q, head_dim)

            # 缩放点积注意力
            scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)
            multi_head_outputs.append(scaled_attention)

        # 将所有头的输出拼接起来
        concat_out = torch.cat(multi_head_outputs, dim=-1)  # (batch_size, seq_len_q, num_heads * head_dim)

        # 通过输出线性层
        out = self.fc_out(concat_out)  # (batch_size, seq_len_q, embed_size)

        return out

def test_shape():
    batch_size = 2
    seq_len = 5
    embed_size = 8
    num_heads = 4

    model = MultiHeadAttention(embed_size, num_heads)

    x = torch.randn(batch_size, seq_len, embed_size)

    out, attn = model(x, x, x)

    print("Output shape:", out.shape)
    print("Number of heads:", len(attn))
    print("Attention shape per head:", attn[0].shape)


test_shape()
import math
import torch
import torch.nn as nn


class selfAttentionV1(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(embed_size, embed_size*3) # 初始化三个投影矩阵
    def forward(self,X):
        # X shape (batch_size, seq_len, embed_size)
        w_q,w_k,w_v = torch.split(self.projection(X),self.embed_size,dim=-1)
        # Q K V shape (batch_size, seq_len, embed_size)
        dim_k=w_q.size(-1)
        # Q (batch_size, seq_len, embed_size) @ K.transpose (batch_size, embed_size, seq_len) -> (batch_size, seq_len, seq_len)
        attention_weight= torch.softmax(torch.matmul(w_q,w_k.transpose(-2,-1))/math.sqrt(dim_k),dim=-1)
        output =attention_weight @ w_v # (batch_size, seq_len, seq_len) @  (batch_size, seq_len, embed_size)
        return output, attention_weight
X =torch.rand(3,5,4)
attention_test1= selfAttentionV1(4)
output,attention_weight= attention_test1(X)
print(output,attention_weight)


class selfAttentionV2(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size= embed_size
        self.projection = nn.Linear(embed_size, embed_size*3)
    def forward(self,X):
        w_q,w_k,w_v= torch.split(self.projection(X), self.embed_size, dim=-1)
        dim_k=w_q.size(-1)
        attention_weight= torch.softmax(torch.matmul(w_q, w_k.transpose(-2,-1))/math.sqrt(dim_k),dim=-1)
        output =torch.matmul(attention_weight,w_v)
        return output, attention_weight

X =torch.rand(3,5,4)
attention_test2= selfAttentionV2(4)
output,attention_weight= attention_test2(X)
print(output,attention_weight)


class selfAttentionV3(nn.Module):
    def __init__(self, embed_dim,dropout_rate=0.1):
        super().__init__()
        self.embed_dim =embed_dim
        self.projection = nn.Linear(embed_dim,embed_dim*3)
        self.dropout=nn.Dropout(dropout_rate)
    def forward(self,X,attention_mask=None):
        w_q,w_k,w_v= torch.split(self.projection(X),self.embed_dim,dim=-1)
        dim_k=w_q.size(-1)
        # (batch, seq, seq)
        attention_weight= torch.matmul(w_q,w_k.transpose(-2,-1))/math.sqrt(dim_k)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask==0,float("-inf"))
        attention_weight= torch.softmax(attention_weight,dim=-1)
        attention_weight= self.dropout(attention_weight)
        output=attention_weight @ w_v
        return output,attention_weight
    
X =torch.rand(3,5,4)
attention_test3= selfAttentionV3(4)
output,attention_weight= attention_test3(X)
print(output,attention_weight)





class SelfAttentionInterview(nn.Module):
    def __init__(self, embed_size: int, dropout_rate:float =0.1)->None:
        super().__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.attention_dropout = nn.Dropout(dropout_rate)
    def forward(self, X, attention_mask=None):
        # X shape is (batch_size, seq_len, embed_size)
        Q =self.query(X)
        K =self.key(X)
        V =self.value(X)

        attention_weight = Q @ K.transpose(-1,-2)/math.sqrt(self.embed_size)

        if attention_mask is not None:
            attention_weight =attention_weight.masked_fill (
                attention_mask==0,
                float("-inf")
            )
        attention_weight=torch.softmax(attention_weight, dim=-1) # 指定在最后一个维度上做softmax
        attention_weight = self.attention_dropout(attention_weight)
        output = attention_weight @ V
        return output, attention_weight

X =torch.rand(3,5,4)
attention_test= SelfAttentionInterview(4)
output,attention_weight= attention_test(X)
print(output,attention_weight)
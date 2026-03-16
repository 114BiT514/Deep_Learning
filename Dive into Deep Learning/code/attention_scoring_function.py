import math
import torch
from torch import nn
from d2l import torch as d2l

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    # None 在索引中表示 在这个位置增加一个维度，以便进行广播操作
    #这个式子得到一个由True和False组成的与X形状相同的张量
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    #~表示按元素取反,这里传入的是一个bool索引，选定所有True的位置
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        #X.shape = (batch_size, num_queries, num_keys)。三维张量
        shape = X.shape
        #valid_lens:1D(batch_size)或2D张量(batch_size,num_queries)
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# valid_lens = torch.tensor([2, 6])
# attention = AdditiveAttention(2, 20, 8, 0.1)
# attention.eval()
# print(attention(queries, keys, values, valid_lens))
# d2l.show_heatmaps(attention.attention_weights.reshape(1, 1, 2, 10), xlabel='Keys', ylabel='Queries')
# d2l.plt.show()

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        #transpose()交换张量第1维和第2维
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# queries = torch.normal(0, 1, (2, 1, 2))
# attention = DotProductAttention(0.5)
# attention.eval()
# print(attention(queries, keys, values, valid_lens))
# d2l.show_heatmaps(attention.attention_weights.reshape(1, 1, 2, 10), xlabel='Keys', ylabel='Queries')
# d2l.plt.show()



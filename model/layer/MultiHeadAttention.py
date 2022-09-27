import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")
        self.n_heads = config.getint("model", "n_heads")
        self.max_cls_len = config.getint("data", "max_cls_len")

        # 强制 hid_dim 必须整除 h
        assert self.hidden_size % self.n_heads == 0
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.do = nn.Dropout(config.getfloat("model", "dropout"))
        # 缩放
        # self.scale = (torch.sqrt(torch.FloatTensor([self.hidden_size // self.n_heads])))

    def forward(self, lstm_output, cls_len):
        batch_size = lstm_output.shape[0]
        Q = self.w_q(lstm_output)
        K = self.w_k(lstm_output)
        V = self.w_v(lstm_output)
        # [batch_size, T, 768]->[batch_size, T, 8, 96]->[batch_size, 8, T, 96]
        Q = Q.view(batch_size, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hidden_size // self.n_heads).permute(0, 2, 1, 3)

        # Q 乘以 K的转置，除以scale  [batch_size, 8, T, 96] * [batch_size, 8, 96, T] -> [batch_size, 8, T, T]
        scale = (torch.sqrt(torch.FloatTensor([self.hidden_size // self.n_heads]))).cuda()
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) 
        attention = attention / scale

        # mask操作， 把 mask 为 1 的位置的 attention 设置为 -inf
        # print(cls_len)
        # print(lstm_output.shape)
        mask_batch = []
        for i in range(batch_size):
            mask_head = []
            for n in range(self.n_heads):
                mask = np.ones((self.max_cls_len, self.max_cls_len))#
                for j in range(cls_len[i].item()):
                    for k in range(cls_len[i].item()):
                        mask[j][k] = 0
                mask_head.append(mask)
            mask_batch.append(mask_head)
        mask_batch = (torch.tensor(np.array(mask_batch), dtype=torch.bool)).cuda()
        # print(mask_batch.shape)
        # print(mask_batch)
        # mask_batch = mask_batch.byte()
        attention = attention.masked_fill(mask_batch, -float('inf'))

        attention = self.do(torch.softmax(attention, dim=-1))

        # attention结果与V相乘，得到多头注意力的结果
        attention_output = torch.matmul(attention, V)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        # 把多组注意力的结果拼接起来
        attention_output = attention_output.view(batch_size, -1, self.n_heads * (self.hidden_size // self.n_heads))
        attention_output = self.fc(attention_output)
        return attention_output

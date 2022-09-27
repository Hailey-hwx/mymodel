import torch.nn as nn
import torch
import math

# 0.mask
class TransMask(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TransMask,self).__init__()
        self.transformer = nn.Transformer()

    def forward(self,x):
        key_padding_mask = torch.zeros(x.size())
        key_padding_mask[x == 0] = -float('inf')
        tgt_mask = self.transformer.generate_square_subsequent_mask(sz=x.size(-1))

        return key_padding_mask, tgt_mask

# 1.文本编码层
class TextEmbedding(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        '''
        :param vocab:  词表的大小
        :param d_model: 词嵌入的维度
        '''
        super(TextEmbedding,self).__init__()
        self.vocab = config.getint("model", "vocab")
        self.d_model = config.getint("model", "hidden_size")
        
        self.lut = nn.Embedding(self.vocab,self.d_model,padding_idx=0)
        

    def forward(self,x):
        '''
        :param x: 输入给模型的文本通过词汇映射后的张量
        :return:
        '''
        return self.lut(x) * math.sqrt(self.d_model)

# 2.位置编码层
class PositionalEnconding(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        '''
        :param d_model: 词嵌入维度
        :param dropout: 丢失率
        :param max_len: 每个句子的最长长度
        '''
        super(PositionalEnconding,self).__init__()
        self.d_model = config.getint("model", "hidden_size")
        self.dropout = config.getfloat("model", "dropout")
        self.max_len = config.getint("data", "max_len")
        # 实例化dropout层
        self.dpot = nn.Dropout(p=self.dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(self.max_len, self.d_model)

        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0, self.max_len).unsqueeze(1)

        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000)/self.d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze((0))

        # 把pe位置编码矩阵注册成模型的buffer
        # 模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''
        :param x: 文本的词嵌入表示
        :return:
        '''
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dpot(x)

class Generator(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        '''
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        '''
        super(Generator, self).__init__()
        self.vocab = config.getint("model", "vocab")
        self.d_model = config.getint("model", "hidden_size")
        self.project = nn.Linear(self.d_model, self.vocab)

    def forward(self,x):
        return self.project(x).softmax(dim=-1)

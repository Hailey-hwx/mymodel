
import torch.nn as nn
from transformers import AutoModel


class LawformerEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LawformerEncoder, self).__init__()
        self.lawformer = AutoModel.from_pretrained('thunlp/Lawformer')

    def forward(self, x):
        y = self.lawformer(x)['last_hidden_state']

        return y

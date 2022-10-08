from model.LCASE import LCASE
from model.baseline.transformer import Transformer
from model.baseline.bert import Bert
from model.baseline.seq2seq import Seq2seq

model_list = {
    "LCASE": LCASE,
    "Transformer":Transformer,
    "Seq2seq":Seq2seq,
    "Bert":Bert
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError

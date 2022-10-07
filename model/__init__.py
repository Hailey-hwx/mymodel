from model.LCASE import LCASE
from model.baseline.transformer import Transformer
from model.baseline.bert import Bert

model_list = {
    "LCASE": LCASE,
    "Transformer":Transformer,
    "Bert":Bert
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError

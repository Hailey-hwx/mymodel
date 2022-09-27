import torch
import torch.nn as nn
from transformers import BertModel, AutoModel


class LawformerCLSEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LawformerCLSEncoder, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lawformer = AutoModel.from_pretrained('thunlp/Lawformer')
        self.max_cls_len = config.getint("data", "max_cls_len")
        self.hidden_size = config.getint("model", "hidden_size")

    def forward(self, input_ids, token_type_ids, attention_mask, cls_ids, cls_len, part):
        if part == 3:
            outputs = self.lawformer(input_ids, token_type_ids, attention_mask)
            last_hidden = outputs[0]

            cls_inputs = []

            # print(last_hidden.shape)
            # print(cls_ids)

            for i, cls_id in enumerate(cls_ids):
                cls_tokens = []
                cls_id = cls_id[:cls_len[i].item()]
                for j in cls_id:
                    cls_tokens.append(last_hidden[i][j])
                if len(cls_tokens) >= self.max_cls_len:
                    cls_tokens = cls_tokens[:self.max_cls_len]
                else:
                    for n in range(len(cls_tokens), self.max_cls_len):
                        cls_tokens.append((torch.LongTensor([0] * self.hidden_size).cuda()))
                # cls_tokens = cls_tokens.cuda()
                cls_tokens = torch.stack(cls_tokens, 0)
                # cls_tokens = cls_tokens.cuda()
                cls_tokens = cls_tokens.unsqueeze(0)
                cls_inputs.append(cls_tokens)

            sen_inputs = cls_inputs[0]
            for i, sen in enumerate(cls_inputs):
                if i == 0:
                    continue
                else:
                    sen_inputs = torch.cat([sen_inputs, sen], dim=0)
        else:
            outputs = self.lawformer(input_ids, attention_mask=attention_mask)
            sen_inputs = outputs[0]

        return sen_inputs


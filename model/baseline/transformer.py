import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
print(BASE_DIR)

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model.decoder.TransDecoder import TransMask, TextEmbedding, PositionalEnconding, Generator


class Transformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Transformer, self).__init__()
        self.mask = TransMask(config, gpu_list, *args, **params)
        self.te = TextEmbedding(config, gpu_list, *args, **params)
        self.pe = PositionalEnconding(config, gpu_list, *args, **params)
        self.ge = Generator(config, gpu_list, *args, **params)
        self.d_model = config.getint("model", "hidden_size")
        self.transformer = nn.Transformer(d_model=self.d_model)
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        if config.getboolean("distributed", "use"):
            self.mask = nn.parallel.DistributedDataParallel(self.mask, device_ids=device)
            self.te = nn.parallel.DistributedDataParallel(self.te, device_ids=device)
            self.pe = nn.parallel.DistributedDataParallel(self.pe, device_ids=device)
            self.ge = nn.parallel.DistributedDataParallel(self.ge, device_ids=device)
            self.transformer = nn.parallel.DistributedDataParallel(self.transformer, device_ids=device)
        else:
            self.mask = nn.DataParallel(self.mask, device_ids=device)
            self.te = nn.DataParallel(self.te, device_ids=device)
            self.pe = nn.DataParallel(self.pe, device_ids=device)
            self.ge = nn.DataParallel(self.ge, device_ids=device)
            self.transformer = nn.DataParallel(self.transformer, device_ids=device)

    def forward(self, data, config, gpu_list, mode):
        x = data

        document_input_ids = x['document_input_ids']
        summary_input_ids = x['summary_input_ids']

        _summary = summary_input_ids
        summary_str = []
        for batch in _summary:
            summary_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            summary_s = ''.join(summary_s)
            summary_str.append(summary_s)
        # print(summary_str)

        if mode != "test":
            # transformer-decoder
            tgt_key_padding_mask, tgt_mask = self.mask(summary_input_ids)
            src_key_padding_mask = torch.zeros(document_input_ids.size())
            src_key_padding_mask[document_input_ids == 0] = -float('inf')
            source_input = self.te(document_input_ids)
            source_input = self.pe(source_input)
            source_input = source_input.transpose(0,1)

            target_out = self.te(summary_input_ids)
            target_out = self.pe(target_out)
            target_out = target_out.transpose(0,1)
            out = self.transformer(src=source_input, tgt=target_out, tgt_mask=tgt_mask, src_key_padding_mask = src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            out = out.transpose(0,1)
            summary_score = self.ge(out)#predict p[(p0,p1,...,p21127)*T]
            summary_score = summary_score.transpose(1,2)
            summary_out = summary_score.argmax(dim=1)#predict textcode

            loss = self.criterion(summary_score, summary_input_ids)

            # print(loss)

            pre_summary_temp = self.tokenizer.batch_decode(summary_out, skip_special_tokens=True)#predict summary
            pre_summary = ["".join(str.replace(" ", "")) for str in pre_summary_temp]

            # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
            return {"loss": loss, "summary": summary_str, "pre_summary": pre_summary}


        # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
        return {"summary": summary_str, "pre_summary": pre_summary}
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目�?
sys.path.append(BASE_DIR)
sys.setrecursionlimit(2000)

import numpy as np
import torch
import torch.nn as nn
from model.encoder.LawformerCLSEncoder import LawformerCLSEncoder
from model.layer.BiLSTM import BiLSTM
from model.layer.MultiHeadAttention import MultiHeadAttention
from model.layer.Classifier import Classifier
from tools.accuracy_init import init_accuracy_function
from transformers import BertTokenizer, AutoModel
from model.decoder.TransDecoder import TransMask, TextEmbedding, PositionalEnconding, Generator


class MyModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MyModel, self).__init__()
        # extractive
        self.lawformer_cls = LawformerCLSEncoder(config, gpu_list, *args, **params)
        self.bilstm = BiLSTM(config, gpu_list, *args, **params)
        self.attention = MultiHeadAttention(config, gpu_list, *args, **params)
        self.classifier = Classifier(config, gpu_list, *args, **params)
        self.max_cls_len = config.getint("data", "max_cls_len")

        # abstractive
        # encoder
        # self.lawformer = AutoModel.from_pretrained('thunlp/Lawformer')
        # decoder
        self.mask = TransMask(config, gpu_list, *args, **params)
        self.te = TextEmbedding(config, gpu_list, *args, **params)
        self.pe = PositionalEnconding(config, gpu_list, *args, **params)
        self.ge = Generator(config, gpu_list, *args, **params)
        self.d_model = config.getint("model", "hidden_size")
        self.n_heads = config.getint("model", "n_heads")
        self.trans_layer = config.getint("model", "trans_layer")
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_heads)    
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.trans_layer)

        self.max_lawformer_len = config.getint('data', 'max_lawformer_len')
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.max_sum_len = config.getint('output', 'max_sum_len')

    def get_summary2(self, source_input_ids, source_attention_mask, target_inputs):
        # lawformer-encoder
        # outputs = self.lawformer(source_input_ids, attention_mask=source_attention_mask)
        # encoder_output = outputs[0]
        encoder_output = self.lawformer_cls(source_input_ids, 0, source_attention_mask, 0, 0, 2)
        # transformer-decoder
        tgt_key_padding_mask, tgt_mask = self.mask(target_inputs)
        target_out = self.te(target_inputs)
        target_out = self.pe(target_out)
        target_out = target_out.transpose(0,1)
        encoder_output = encoder_output.transpose(0,1)
        out = self.transformer_decoder(tgt=target_out, memory=encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = out.transpose(0,1)
        part2_score = self.ge(out)#predict p[(p0,p1,...,p21127)*T]
        part2_score = part2_score.transpose(1,2)
        part2_out = part2_score.argmax(dim=1)#predict textcode
        # print(part2_out.shape)
        # part2_summary = self.tokenizer.batch_decode(part2_out, skip_special_tokens=True)#predict summary
        # part2_summary = ["".join(str.replace(" ", "")) for str in part2_summary]
        return part2_score, part2_out

    def get_summary2_test(self, source_input_ids, source_attention_mask):
        # lawformer-encoder
        # outputs = self.lawformer(source_input_ids, attention_mask=source_attention_mask)
        # encoder_output = outputs[0]
        encoder_output = self.lawformer_cls(source_input_ids, 0, source_attention_mask, 0, 0, 2)
        # transformer-decoder
        target_inputs = (torch.LongTensor([[101]])).cuda()

        ######循环  直到输出102  或者长度超过1600
        for i in range(self.max_sum_len):
            target_out = self.te(target_inputs)
            target_out = self.pe(target_out)
            target_out = target_out.transpose(0,1)

            encoder_output = encoder_output.transpose(0,1)
            out = self.transformer_decoder(tgt=target_out, memory=encoder_output)
            out = out.transpose(0,1)
            part2_score = self.ge(out)#predict p[(p0,p1,...,p21127)*T]
            part2_score = part2_score.transpose(1,2)
            part2_out = part2_score.argmax(dim=1)#predict textcode
            predict = part2_out[:,-1].unsqueeze(1)
            target_inputs = torch.cat([target_inputs, predict], dim=1)

            if predict.equal(torch.LongTensor([[102]])):
                break

        # part2_summary = self.tokenizer.batch_decode(part2_out, skip_special_tokens=True)#predict summary
        # part2_summary = ["".join(str.replace(" ", "")) for str in part2_summary]
        return part2_score, part2_out

    def get_summary3(self, part3_input_ids, part3_token_type_ids, part3_attention_mask, part3_cls_ids, part3_cls_len, part3_all):
        part3_sen_input = self.lawformer_cls(part3_input_ids, part3_token_type_ids, part3_attention_mask,
                                                   part3_cls_ids, part3_cls_len, 3)
        # print(part3_sen_input.shape)
        part3_lstm_output = self.bilstm(part3_sen_input, part3_cls_len)
        # print(part3_lstm_output.shape)
        part3_attention_output = self.attention(part3_lstm_output, part3_cls_len)
        part3_score = self.classifier(part3_attention_output)
        # print(part3_score.dtype)
        # 删除nan
        part3_score_list = []
        for batch in part3_score:
            part3_score_list_batch = []
            for score in batch:
                if torch.isnan(score).any():
                    part3_score_list_batch.append((torch.LongTensor([1,0])).cuda())
                else:
                    part3_score_list_batch.append(score)
            if len(part3_score_list_batch) < self.max_cls_len:
                for n in range(len(part3_score_list_batch), self.max_cls_len):
                    part3_score_list_batch.append((torch.LongTensor([1,0])).cuda())
            part3_score_list_batch = torch.stack(part3_score_list_batch, 0)
            part3_score_list_batch = part3_score_list_batch.unsqueeze(0)
            part3_score_list.append(part3_score_list_batch)
        part3_score = part3_score_list[0]
        # print(part3_score.dtype)
        # print(part3_score)
        for i, sen in enumerate(part3_score_list):
            if i == 0:
                continue
            else:
                part3_score = torch.cat([part3_score, sen], dim=0)#predict p[(p0,p1)*T]
        part3_out = part3_score.argmax(dim=2)
        return part3_score, part3_out
    
    def forward(self, data, mode):
        # extractive
        # part1_all = data['part1_all']
        # print(part1_all)

        part3_input_ids = data['input3_input_ids']
        part3_token_type_ids = data['input3_token_type_ids']
        part3_attention_mask = data['input3_attention_mask']
        part3_cls_ids = data['input3_cls_ids']
        part3_cls_len = data['input3_cls_len']
        part3_all = data['part3_all']

        # abstractive
        source_input_ids = data['input2_input_ids']
        source_attention_mask = data['input2_attention_mask']
        target_inputs = data['output2_input_ids']# train与test的target_input不同

        # part1  extractive
        # part1_summary = self.get_summary1(part1_all)

        # part3  extractive
        part3_score, part3_out = self.get_summary3(part3_input_ids, part3_token_type_ids, part3_attention_mask, part3_cls_ids, part3_cls_len, part3_all)
        # print(part3_str)

        # part2
        if mode != "test":
            part2_score, part2_out = self.get_summary2(source_input_ids, source_attention_mask, target_inputs)
        else:
            part2_score, part2_out = self.get_summary2_test(source_input_ids, source_attention_mask, target_inputs)
        
        return {
                'part2_score': part2_score, 'part2_out': part2_out,
                'part3_score': part3_score, 'part3_out': part3_out
                }


class LCASE(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LCASE, self).__init__()
        self.LCASE = MyModel(config, gpu_list, *args, **params)
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.criterion = nn.CrossEntropyLoss()
        # self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        if config.getboolean("distributed", "use"):
            self.LCASE = nn.parallel.DistributedDataParallel(self.LCASE, device_ids=device)
        else:
            self.LCASE = nn.DataParallel(self.LCASE, device_ids=device)

    def do_label(self, summary, document):
        y = [0] * len(document)
        document_start = 0
        for i in range(0, len(summary)):
            if summary[i] in document[document_start:]:
                _document_start = document[document_start:].index(summary[i])
                y[document_start + _document_start] = 1
                document_start = document_start +  _document_start + 1
        return y 

    def get_extractive_label(self, output3, part3_str, part3_score):
        output3_str = []
        for batch in output3:
            output3_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            output3_s = ["".join(str.replace(" ", "")) for str in output3_s]
            output3_str.append(output3_s)

        ex_label = []
        for batch in range(len(output3_str)): #batch_size
            label = self.do_label(output3_str[batch], part3_str[batch])
            if len(label) >= part3_score.shape[1]:
                label = label[:part3_score.shape[1]]
            else:
                for i in range(part3_score.shape[1]-len(label)):
                    label.append(0)
            ex_label.append(label)
        ex_label = (torch.tensor(np.array(ex_label), dtype=torch.int64)).cuda()
        return ex_label

    def get_summary1(self, part1_all):
        part1_str = []
        for batch in part1_all:
            part1_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            part1_s = ["".join(str.replace(" ", "")) for str in part1_s]
            part1_str.append(part1_s)

        user_dict = []
        with open('./data/user_dict.txt', encoding='utf-8') as f:
            for l in f:
                w = l.split()[0]
                user_dict.append(w)  
        part1_s = []
        for batch in part1_str:
            s = []
            for i in batch:
                if i in user_dict:
                    s.append(i)
            part1_s.append(s)
        part1_summary = []
        for batch in part1_s:
            summary = '原被告系'+batch[0]+'关系'
            part1_summary.append(summary)# predict summary
        return part1_summary

    def forward(self, data, config, gpu_list, mode):
        x = data

        y = self.LCASE(x, mode)
        part1_summary = self.get_summary1(x['part1_all'])
        # part1_summary = y['part1_summary']
        part2_score = y['part2_score']
        part2_out = y['part2_out']
        part2_summary_temp = self.tokenizer.batch_decode(part2_out, skip_special_tokens=True)#predict summary
        part2_summary = ["".join(str.replace(" ", "")) for str in part2_summary_temp]

        part3_score = y['part3_score']
        # print(part3_score.dtype)
        part3_out = y['part3_out']
        part3_all = x['part3_all']
        part3_str = []
        for batch in part3_all:
            part3_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            part3_s = ["".join(str.replace(" ", "")) for str in part3_s]
            part3_str.append(part3_s)
        part3_summary = []
        for i, batch in enumerate(part3_out):
            out_batch = []
            for j, predict in enumerate(batch):
                if predict.item() == 1:
                    out_batch.append(part3_str[i][j])
            out_batch = ''.join(out_batch)
            part3_summary.append(out_batch)#predict summary

        pre_summary = []
        summary_str = []
        
        summary = x['summary']
        # print(summary.shape)
        # print(summary)
        for batch in summary:
            summary_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            summary_s = ''.join(summary_s)
            summary_str.append(summary_s)

        if mode != "test":
            print(part1_summary)
            print(part2_summary)
            print(part3_summary)
            output3 = x['output3']
            ab_label = x['output2_input_ids']
            ex_label = self.get_extractive_label(output3, part3_str, part3_score)
            
            part3_score = part3_score.transpose(1,2)
            part3_score.to(torch.float32)
            print(part3_score.dtype)
            print(ex_label.dtype)
            print(part2_score.dtype)
            print(ab_label.dtype)
            ex_loss = self.criterion(part3_score, ex_label)
            # print(ex_loss)

            # part2_score = part2_score.transpose(1,2)
            ab_loss = self.criterion(part2_score, ab_label)
            # print(ab_loss)

            loss = (ex_loss + ab_loss) / 2
            print(loss)

            for i in range(len(part1_summary)):
                s = part1_summary[i] + part2_summary[i] + part3_summary[i]
                pre_summary.append(s)

            # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
            return {"loss": loss, "summary": summary_str, "pre_summary": pre_summary}


        for i in range(len(part1_summary)):
                s = part1_summary[i] + part2_summary[i] + part3_summary[i]
                pre_summary.append(s)

        # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
        return {"summary": summary_str, "pre_summary": pre_summary}

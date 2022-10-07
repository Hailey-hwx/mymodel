import json
import jieba
from rouge import Rouge
import torch
import math
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from DucomentSegment import DocumentSegment
from SummarySegment import SummarySegment
from LawformerFormatter import LawformerFormatter
from SummaryFormatter import SummaryFormatter

def process(data, mode='train'):
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        document_all = []
        idx = []
        summary_all = []

        for temp_data in data:
            idx.append(temp_data["id"])
            document = []
            for text in temp_data["text"]:
                document.append(text["sentence"])
            document = ''.join(document)

            if mode != "test":
                summary = temp_data["summary"]
                document_all.append(document)
                summary_all.append(summary)
            else:
                document_all.append(document)
        
        document_token = tokenizer(document_all, max_length=3000, truncation=True, padding=True, return_tensors='pt')
        document_input_ids = document_token['input_ids']

        if mode != "test":
            summary_token = tokenizer(summary_all, max_length=3000, truncation=True, padding=True, return_tensors='pt')
            summary_input_ids = summary_token['input_ids']
            return {
                    'document_input_ids': document_input_ids,
                    'summary_input_ids': summary_input_ids
                    }
        else:
            return {
                    'document_input_ids': document_input_ids
                    }




# def process(data, mode='train'):
#     document_all = []
#     input_input_ids = []
#     input_token_type_ids = []
#     input_attention_mask = []
#     input_cls_ids = []
#     input_cls_len = []
#     idx = []
#     summary_all = []
#     _summary = []
#     tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

#     for temp_data in data:
#         # print(temp_data)
#         idx.append(temp_data["id"])
#         document = []
#         for text in temp_data["text"]:
#             document.append(text["sentence"])
#         document = document[10:-10]
#         # print(document)
#         document = ''.join(document)
#         document_segment = DocumentSegment()
#         document = document_segment.sentence_split(document)
#         f_document = []
#         for s in document:
#             if s != '':
#                 f_document.append(s)
#         # print(f_document)

#         if mode != 'test':
#             summary = document_segment.sentence_split(temp_data["summary"])
#             f_summary = []
#             for s in summary:
#                 if s != '':
#                     f_summary.append(s)
#             # print(f_summary)

#             # 没分词
#             _summary.append(temp_data["summary"])

#             document_token = tokenizer(f_document, max_length=512, truncation=True)
#             document_all.append(document_token['input_ids'])

#             summary_token = tokenizer(f_summary, max_length=512, truncation=True)
#             summary_all.append(summary_token['input_ids'])

#             # print(document_token)

#             bert_formatter = LawformerFormatter(512)
#             input_ids, token_type_ids, attention_mask, cls_ids, cls_len = bert_formatter.cls_process(document_token)

#             input_input_ids.append(input_ids)
#             input_token_type_ids.append(token_type_ids)
#             input_attention_mask.append(attention_mask)
#             input_cls_ids.append(cls_ids)
#             input_cls_len.append(cls_len)

#         else:
#             document_token = tokenizer(f_document, max_length=512, truncation=True)
#             document_all.append(document_token['input_ids'])

#             _summary.append(temp_data["summary"])

#             bert_formatter = LawformerFormatter(512)
#             input_ids, token_type_ids, attention_mask, cls_ids, cls_len = bert_formatter.cls_process(document_token)

#             input_input_ids.append(input_ids)
#             input_token_type_ids.append(token_type_ids)
#             input_attention_mask.append(attention_mask)
#             input_cls_ids.append(cls_ids)
#             input_cls_len.append(cls_len)

#     input_input_ids = torch.LongTensor(input_input_ids)
#     input_token_type_ids = torch.LongTensor(input_token_type_ids)
#     input_attention_mask = torch.LongTensor(input_attention_mask)
#     input_cls_ids = torch.LongTensor(input_cls_ids)
#     input_cls_len = torch.LongTensor(input_cls_len)

#     summary_formatter = SummaryFormatter()

#     document_all_list = summary_formatter.summary_process(document_all)
#     document_all = torch.LongTensor(document_all_list)

#     _summary_token = tokenizer(_summary, max_length=512, truncation=True, padding=True, return_tensors='pt')
#     _summary = _summary_token['input_ids']

#     if mode != "test":
#         summary_all_list = summary_formatter.summary_process(summary_all)
#         summary_all = torch.LongTensor(summary_all_list)
#         return {
#                 'document_all': document_all,
#                 'input_input_ids': input_input_ids,
#                 'input_token_type_ids': input_token_type_ids,
#                 'input_attention_mask': input_attention_mask,
#                 'input_cls_ids': input_cls_ids,
#                 'input_cls_len': input_cls_len,
#                 'summary_all': summary_all,
#                 '_summary': _summary
#                 }
#     else:
#         return {
#                 'document_all': document_all,
#                 'input_input_ids': input_input_ids,
#                 'input_token_type_ids': input_token_type_ids,
#                 'input_attention_mask': input_attention_mask,
#                 'input_cls_ids': input_cls_ids,
#                 'input_cls_len': input_cls_len,
#                 '_summary': _summary
#                 }

def load_user_dict(filename):
        user_dict = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                w = l.split()[0]
                user_dict.append(w)
        return user_dict


data = []
with open('./data/train_temp.json', encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
print(len(data))

# jieba.load_userdict('./data/user_dict.txt')
# jieba.initialize()

# data processing document segment
loader = torch.utils.data.DataLoader(dataset=data,
                                     batch_size=2,
                                     collate_fn=process,
                                     shuffle=True,
                                     drop_last=True)
print(len(loader))
for i, item in enumerate(loader):
    # print(item)
    break

document_input_ids = item['document_input_ids']
summary_input_ids = item['summary_input_ids']
# print(document_input_ids)
# print(document_input_ids.shape)
# print(summary_input_ids)
# print(summary_input_ids.shape)
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        

vocab = 21128
d_model = 768
dropout = 0.1

# seq2seq
print(document_input_ids.shape)
print(document_input_ids)
print(summary_input_ids.shape)

# encoder
doc_len = []
for batch in document_input_ids:
    len = 0
    for i in batch:
        if i != 0:
            len=len+1
    doc_len.append(len)
doc_len = torch.LongTensor(doc_len)
print(doc_len)

embedding = nn.Embedding(vocab, d_model)
# self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
lstm = nn.LSTM(768, 768, bidirectional=True)
fc = nn.Linear(768 * 2, 768)
dropout = nn.Dropout(dropout)

embedded = dropout(embedding(document_input_ids))
packed_embedded = rnn_utils.pack_padded_sequence(embedded, doc_len, batch_first=True, enforce_sorted=False)
lstm.flatten_parameters()
packed_outputs, hidden = lstm(packed_embedded) 
outputs, length = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)#, , total_length=3000
hidden = torch.tanh(fc(torch.cat((hidden[0][:, -2, :], hidden[0][:, -1, :]), dim=1)))
# print(outputs.shape)  #[batchsize, doclen(), 768*2]
# print(hidden.shape)    #[batchsize,768]

# attention
attn = nn.Linear((768 * 2) + 768, 768)
v = nn.Linear(768, 1, bias=False)
encoder_outputs = outputs
batch_size = encoder_outputs.shape[0]
doc_len = encoder_outputs.shape[1]
 
hidden = hidden.unsqueeze(1).repeat(1, doc_len, 1)
# print(hidden.shape)
# print(outputs.shape)
# hidden = [batch size, doc len, dec hid dim]
# outputs = [batch size, doc len, enc hid dim * 2]
 
energy = torch.tanh(attn(torch.cat((hidden, encoder_outputs), dim=2)))
# energy = [batch size, doc len, dec hid dim]
 
attention = v(energy).squeeze(2)
# attention = [batch size, doc len]
mask = (document_input_ids != 0)
attention = attention.masked_fill(mask == 0, -1e10)
a = attention.softmax(dim=1)
print(a.shape)

# decoder
sum_len = summary_input_ids.shape[1]
decoder_outputs = torch.zeros(batch_size, sum_len, vocab)###
input = summary_input_ids[:,0]
# print(input)
for t in range(1,sum_len):
    # output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)













# tgt_key_padding_mask = torch.zeros(summary_input_ids.size())
# tgt_key_padding_mask[summary_input_ids == 0] = -float('inf')
# src_key_padding_mask = torch.zeros(document_input_ids.size())
# src_key_padding_mask[document_input_ids == 0] = -float('inf')
# transformer = nn.Transformer()
# tgt_mask = transformer.generate_square_subsequent_mask(sz=summary_input_ids.size(-1))

# lut = nn.Embedding(vocab,d_model,padding_idx=0)
# target_out = lut(summary_input_ids) * math.sqrt(d_model)

# source_input = lut(document_input_ids) * math.sqrt(d_model)

# dpot = nn.Dropout(p=dropout)
# pe = torch.zeros(3000, d_model)
# position = torch.arange(0, 3000).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))
# pe[:,0::2] = torch.sin(position * div_term)
# pe[:,1::2] = torch.cos(position * div_term)
# pe = pe.unsqueeze((0))
# target_out = target_out + pe[:, : target_out.size(1)].requires_grad_(False)
# target_out = dpot(target_out)

# source_input = source_input + pe[:, : source_input.size(1)].requires_grad_(False)
# source_input = dpot(source_input)

# transformer = nn.Transformer(d_model=768)

# target_out = target_out.transpose(0,1)
# source_input = source_input.transpose(0,1)

# out = transformer(src=source_input, tgt=target_out, tgt_mask=tgt_mask, src_key_padding_mask = src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
# out = out.transpose(0,1)
# print(out.shape)


# # G = Generator(d_model, vocab)
# project = nn.Linear(d_model, vocab)
# out = project(out).softmax(dim=-1)
# out = out.transpose(1,2)
# summary_out = out.argmax(dim=1)

# # target_inputs = target_inputs.transpose(1,2)
# print(out.shape)
# # print(target_inputs.shape)
# criterion = torch.nn.CrossEntropyLoss()
# l1 = criterion(out, summary_input_ids)
# print(l1)

# pre_summary_temp = tokenizer.batch_decode(summary_out, skip_special_tokens=True)#predict summary
# pre_summary = ["".join(str.replace(" ", "")) for str in pre_summary_temp]
# print(pre_summary)

# # # out = out.transpose(1,2)
# # out = out.argmax(dim=1)
# # # print(out.shape)
# # print(out.shape)











# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# input_ids = item['input_input_ids']
# # print(input_input_ids.shape)
# token_type_ids = item['input_token_type_ids']
# attention_mask = item['input_attention_mask']
# cls_ids = item['input_cls_ids']
# cls_len = item['input_cls_len']

# bert = BertModel.from_pretrained('bert-base-chinese')

# outputs = bert(input_ids, token_type_ids, attention_mask)
# last_hidden = outputs[0]

# cls_inputs = []

# for i, cls_id in enumerate(cls_ids):
#     cls_tokens = []
#     cls_id = cls_id[:cls_len[i].item()]
#     for j in cls_id:
#         cls_tokens.append(last_hidden[i][j])
#     if len(cls_tokens) >= 200:
#         cls_tokens = cls_tokens[:200]
#     else:
#         for n in range(len(cls_tokens), 200):
#             cls_tokens.append((torch.LongTensor([0] * 768)))
#     cls_tokens = torch.stack(cls_tokens, 0)
#     cls_tokens = cls_tokens.unsqueeze(0)
#     cls_inputs.append(cls_tokens)

# sen_inputs = cls_inputs[0]
# for i, sen in enumerate(cls_inputs):
#     if i == 0:
#         continue
#     else:
#         sen_inputs = torch.cat([sen_inputs, sen], dim=0)

# # print(sen_inputs)
# print(sen_inputs.shape)

# fc = nn.Linear(768, 2)
# h = fc(sen_inputs)
# sent_scores = h.softmax(dim=2)
# # print(sent_scores)
# # print(sent_scores.shape)
# sent_out = sent_scores.argmax(dim=2)
# # print(sent_out)
# print(sent_out.shape)

# document_all = item['document_all']
# document_str = []
# for batch in document_all: 
#     document_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     document_s = ["".join(str.replace(" ", "")) for str in document_s]
#     document_str.append(document_s)
# # print(document_str)
# document_summary = []
# for i, batch in enumerate(sent_out):
#     out_batch = []
#     for j, predict in enumerate(batch):
#         if predict.item() == 1:
#             out_batch.append(document_str[i][j])
#     out_batch = ''.join(out_batch)
#     document_summary.append(out_batch)#predict summary
# print(document_summary)

# _summary = item['_summary']
# summary_str = []
# for batch in _summary:
#     summary_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     summary_s = ''.join(summary_s)
#     summary_str.append(summary_s)
# print(summary_str)

# def do_label(summary, document):
#     y = [0] * len(document)
#     document_start = 0
#     for i in range(0, len(summary)):
#         if summary[i] in document[document_start:]:
#             _document_start = document[document_start:].index(summary[i])
#             y[document_start + _document_start] = 1
#             document_start = document_start +  _document_start + 1
#     return y 

# def get_extractive_label(output3, part3_str, part3_score):
#     output3_str = []
#     for batch in output3:
#         output3_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#         output3_s = ["".join(str.replace(" ", "")) for str in output3_s]
#         output3_str.append(output3_s)

#     ex_label = []
#     for batch in range(len(output3_str)): #batch_size
#         label = do_label(output3_str[batch], part3_str[batch])
#         if len(label) >= part3_score.shape[1]:
#             label = label[:part3_score.shape[1]]
#         else:
#             for i in range(part3_score.shape[1]-len(label)):
#                 label.append(0)
#         ex_label.append(label)
#     ex_label = torch.tensor(np.array(ex_label), dtype=torch.int64)
#     return ex_label

# summary_all = item['summary_all']
# label = get_extractive_label(summary_all, document_str, sent_scores)

# sent_scores = sent_scores.transpose(1,2)
# criterion = nn.CrossEntropyLoss()
# loss = criterion(sent_scores, label)
# print(loss)

# rouge  = Rouge()
# def get_rouge(pre_summary, summary):
#     """计算rouge-1、rouge-2、rouge-l
#     """
#     # pre_summary = [" ".join(jieba.cut(pre.replace(" ", ""))) for pre in pre_summary]
#     # summary = [" ".join(jieba.cut(lab.replace(" ", ""))) for lab in summary]

#     pre_summary = [" ".join(pre.replace(" ", "")) for pre in pre_summary]
#     summary = [" ".join(lab.replace(" ", "")) for lab in summary]
#     try:
#         scores = rouge.get_scores(hyps=pre_summary, refs=summary)
#         return {
#             'rouge-1': scores[0]['rouge-1']['f'],
#             'rouge-2': scores[0]['rouge-2']['f'],
#             'rouge-l': scores[0]['rouge-l']['f'],
#         }
#     except ValueError:
#         return {
#             'rouge-1': 0.0,
#             'rouge-2': 0.0,
#             'rouge-l': 0.0,
#         }

# print(get_rouge(document_summary, summary_str))

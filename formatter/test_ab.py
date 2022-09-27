import math
import sys,os
from unittest import makeSuite
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import json
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoModel
from DucomentSegment import DocumentSegment
from SummarySegment import SummarySegment
from LawformerFormatter import LawformerFormatter
# from model.decoder.TransDecoder import TextEmbedding, PositionalEnconding, Generator

def process(data, mode='train'):
    part1_all = []
    # input1_input_ids = []
    # input1_token_type_ids = []
    # input1_attention_mask = []
    # input1_cls_ids = []
    part2_all = []
    # input2 = []
    part3_all = []
    input3_input_ids = []
    input3_token_type_ids = []
    input3_attention_mask = []
    input3_cls_ids = []
    idx = []
    summary1_all = []
    summary2_all = []
    summary3_all = []
    # output1 = []
    # output2_input_ids = []
    # output2_attention_mask = []
    # output3 = []
    tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # tokenizer2 = BertTokenizer.from_pretrained('bert-base-chinese')

    for temp_data in data:
        # print(temp_data)
        idx.append(temp_data["id"])
        document = []
        for text in temp_data["text"]:
            document.append(text["sentence"])
        document_segment = DocumentSegment()
        part1, part2, part3 = document_segment.get_part(document)
        # 将多个句子合并为一段文本
        part2 = ''.join(part2)

        if mode != 'test':
            summary_segment = SummarySegment()
            summary1, summary2, summary3 = summary_segment.get_part(temp_data["summary"])
            summary1 = ''.join(summary1)
            summary2 = ''.join(summary2)
            summary3 = ''.join(summary3)
            # 删除空数据
            if len(part1) and len(part2) and len(part3) and len(summary1) and len(summary2) and len(summary3):
                part1_all.append(part1)
                part2_all.append(part2)
                part3_all.append(part3)
                summary1_all.append(summary1)
                summary2_all.append(summary2)
                summary3_all.append(summary3)
                # summary.append(temp_data["summary"])

                # part1_token = tokenizer1(part1, max_length=512, truncation=True)
                # part2_token = tokenizer1(part2, max_length=4096, truncation=True, padding=True)
                part3_token = tokenizer1(part3, max_length=4096, truncation=True)

                lawformer_formatter = LawformerFormatter(1000)
                # input_ids1, token_type_ids1, attention_mask1, cls_ids1 = lawformer_formatter.cls_process(part1_token)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = lawformer_formatter.cls_process(part3_token)

                # input1_input_ids.append(input_ids1)
                # input1_token_type_ids.append(token_type_ids1)
                # input1_attention_mask.append(attention_mask1)
                # input1_cls_ids.append(cls_ids1)

                # input2.append(part2_token['input_ids'])

                input3_input_ids.append(input_ids3)
                input3_token_type_ids.append(token_type_ids3)
                input3_attention_mask.append(attention_mask3)
                input3_cls_ids.append(cls_ids3)

                # output.append(summary['input_ids'])

            else:
                break

        else:
            if len(part1) and len(part2) and len(part3):
                part1_all.append(part1)
                part2_all.append(part2)
                part3_all.append(part3)

                # part1_token = tokenizer2(part1, max_length=512, truncation=True)
                # part2_token = tokenizer1(part2, max_length=4096, truncation=True, padding=True)
                part3_token = tokenizer1(part3, max_length=4096, truncation=True)

                lawformer_formatter = LawformerFormatter(512)
                # input_ids1, token_type_ids1, attention_mask1, cls_ids1 = bert_formatter.cls_process(part1_token)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = lawformer_formatter.cls_process(part3_token)

                # input1_input_ids.append(input_ids1)
                # input1_token_type_ids.append(token_type_ids1)
                # input1_attention_mask.append(attention_mask1)
                # input1_cls_ids.append(cls_ids1)

                # input2.append(part2_token)

                input3_input_ids.append(input_ids3)
                input3_token_type_ids.append(token_type_ids3)
                input3_attention_mask.append(attention_mask3)
                input3_cls_ids.append(cls_ids3)

            else:
                break
    # input1_input_ids = torch.LongTensor(input1_input_ids)
    # input1_token_type_ids = torch.LongTensor(input1_token_type_ids)
    # input1_attention_mask = torch.LongTensor(input1_attention_mask)

    input3_input_ids = torch.LongTensor(input3_input_ids)
    input3_token_type_ids = torch.LongTensor(input3_token_type_ids)
    input3_attention_mask = torch.LongTensor(input3_attention_mask)

    if mode != "test":
        part2_token = tokenizer1(part2_all, max_length=4096, truncation=True, padding=True, return_tensors='pt')
        # summary1_token = tokenizer2(summary1_all, max_length=512, truncation=True, padding=True)
        summary2_token = tokenizer1(summary2_all, max_length=4096, truncation=True, padding=True, return_tensors='pt')
        # summary3_token = tokenizer2(summary3_all, max_length=512, truncation=True, padding=True)
        input2_input_ids = part2_token['input_ids']
        input2_attention_mask = part2_token['attention_mask']
        # output1 = summary1_token['input_ids']
        output2_input_ids = summary2_token['input_ids']
        # output2_attention_mask = summary2_token['attention_mask']
        # output3 = summary3_token['input_ids']
        return {
                'part1_all': part1_all,
                # 'input1_input_ids': input1_input_ids,
                # 'input1_token_type_ids': input1_token_type_ids,
                # 'input1_attention_mask': input1_attention_mask,
                # 'input1_cls_ids': input1_cls_ids,
                'input2_input_ids': input2_input_ids,
                'input2_attention_mask': input2_attention_mask,
                'part3_all': part3_all,
                'input3_input_ids': input3_input_ids,
                'input3_token_type_ids': input3_token_type_ids,
                'input3_attention_mask': input3_attention_mask,
                'input3_cls_ids': input3_cls_ids,
                # 'output1': output1,
                'output2_input_ids': output2_input_ids,
                # 'output2_attention_mask': output2_attention_mask,
                'output3': summary3_all
                }
    else:
        part2_token = tokenizer1(part2_all, max_length=4096, truncation=True, padding=True, return_tensors='pt')
        input2_input_ids = part2_token['input_ids']
        input2_attention_mask = part2_token['attention_mask']
        return {
                'part1_all': part1_all,
                # 'input1_input_ids': input1_input_ids,
                # 'input1_token_type_ids': input1_token_type_ids,
                # 'input1_attention_mask': input1_attention_mask,
                # 'input1_cls_ids': input1_cls_ids,
                'input2_input_ids': input2_input_ids,
                'input2_attention_mask': input2_attention_mask,
                'part3_all': part3_all,
                'input3_input_ids': input3_input_ids,
                'input3_token_type_ids': input3_token_type_ids,
                'input3_attention_mask': input3_attention_mask,
                'input3_cls_ids': input3_cls_ids
                }




data = []
with open('./data/train_temp.json', encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
print(len(data))

# data processing document segment
loader = torch.utils.data.DataLoader(dataset=data,
                                     batch_size=3,
                                     collate_fn=process,
                                     shuffle=True,
                                     drop_last=True)
print(len(loader))
for i, item in enumerate(loader):
    # print(item)
    break
print(item['input2_input_ids'])
# print(item['input2_input_ids'].shape)

# lawformer-encoder
lawformer = AutoModel.from_pretrained('thunlp/Lawformer')

source_input_ids = item['input2_input_ids']
source_attention_mask = item['input2_attention_mask']
outputs = lawformer(source_input_ids, attention_mask=source_attention_mask)
encoder_output = outputs[0]
print(source_input_ids.shape[0])
#tensor([[0]])
# target_inputs = torch.LongTensor([[101,134],[101,134],[101,134]])
# print(encoder_output)

# transformer-decoder
vocab = 21128
d_model = 768
dropout = 0.1
target_inputs = item['output2_input_ids']
# print(target_inputs)

tgt_key_padding_mask = torch.zeros(target_inputs.size())
tgt_key_padding_mask[target_inputs == 0] = -float('inf')
# print(tgt_key_padding_mask)
transformer = nn.Transformer()
tgt_mask = transformer.generate_square_subsequent_mask(sz=target_inputs.size(-1))
# print(tgt_mask)

# target_attention_mask = item['output2_attention_mask']
# TE = TextEmbedding(vocab, d_model)
lut = nn.Embedding(vocab,d_model,padding_idx=0)
# target_out = TE(target_inputs)
target_out = lut(target_inputs) * math.sqrt(d_model)


# PE = PositionalEnconding(d_model, dropout, max_len=4096)
# target_out = PE(target_out)
dpot = nn.Dropout(p=dropout)
pe = torch.zeros(4096, d_model)
position = torch.arange(0, 4096).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))
pe[:,0::2] = torch.sin(position * div_term)
pe[:,1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze((0))
target_out = target_out + pe[:, : target_out.size(1)].requires_grad_(False)
target_out = dpot(target_out)
# print(target_out)


decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)

transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

target_out = target_out.transpose(0,1)
encoder_output = encoder_output.transpose(0,1)
print(target_out.shape)
print(encoder_output.shape)

out = transformer_decoder(target_out, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
print(out.shape)
out = out.transpose(0,1)


# G = Generator(d_model, vocab)
project = nn.Linear(d_model, vocab)
out = project(out).softmax(dim=-1)

#jisuan loss
out = out.transpose(1,2)

# target_inputs = target_inputs.transpose(1,2)
print(out.shape)
print(target_inputs.shape)
criterion = torch.nn.CrossEntropyLoss()
l1 = criterion(out, target_inputs)
print(l1)

# out = out.transpose(1,2)
out = out.argmax(dim=1)
# print(out.shape)
print(out.shape)
# print(out[:,-1].unsqueeze(1))#-----new predict
# print(torch.cat([target_inputs, out[:,-1].unsqueeze(1)], dim=1))#------new target_inputs
# print(source_input_ids)

# if target_inputs.equal(torch.LongTensor([[101,134],[101,134],[101,134]])):
#     print('true')

# print(out)
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# output_str = tokenizer.batch_decode(out, skip_special_tokens=True)
# output_str = ["".join(str.replace(" ", "")) for str in output_str]
# print(output_str)






# lawformer2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("thunlp/Lawformer", "bert-base-chinese")

# # outputs = lawformer(input2)
# # last_hidden = outputs[0]
# # print(last_hidden)
# # print(last_hidden.shape)

# # Set tokenizer
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# tokenizer.bos_token = tokenizer.cls_token
# tokenizer.eos_token = tokenizer.sep_token
# # Set model's config
# lawformer2bert.config.decoder_start_token_id = tokenizer.bos_token_id
# lawformer2bert.config.eos_token_id = tokenizer.eos_token_id
# lawformer2bert.config.pad_token_id = tokenizer.pad_token_id

# print(input2_input_ids.shape)
# output = lawformer2bert.generate(input2_input_ids, attention_mask=input2_attention_mask, max_length=128)
# output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
# print(output)
# print(output_str)

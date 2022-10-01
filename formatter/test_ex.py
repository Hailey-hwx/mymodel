# from cProfile import label
import json
import jieba
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoModel
from DucomentSegment import DocumentSegment
from SummarySegment import SummarySegment
from LawformerFormatter import LawformerFormatter
from SummaryFormatter import SummaryFormatter

def process(data, mode='train'):
    part1_all = []
    part2_all = []
    part3_all = []
    input3_input_ids = []
    input3_token_type_ids = []
    input3_attention_mask = []
    input3_cls_ids = []
    input3_cls_len = []
    idx = []
    summary2_all = []
    summary3_all = []
    summary = []
    tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    for temp_data in data:
        # print(temp_data)
        idx.append(temp_data["id"])
        document = []
        for text in temp_data["text"]:
            document.append(text["sentence"])
        document_segment = DocumentSegment()
        part1, part2, part3 = document_segment.get_part(document)
        # print(part1)
        # print(part2)
        # print(part3)
        # print('\n')
        # 将多个句子合并为一段文本
        part2 = ''.join(part2)

        if mode != 'test':
            summary_segment = SummarySegment()
            summary1, summary2, summary3 = summary_segment.get_part(temp_data["summary"])
            print(summary1)
            print(summary2)
            print(summary3)
            print('\n')
            summary1 = ''.join(summary1)
            summary2 = ''.join(summary2)
            summary3 = ''.join(summary3)
            # print(summary1)
            # print(summary2)
            # print(summary3)
            # print('\n')
            summary3 = document_segment.sentence_split(summary3)
            # print(summary3)
            f_summary3 = []
            for s in summary3:
                if s != '':
                    f_summary3.append(s)
            # 删除空数据
            # if len(part1) and len(part2) and len(part3) and len(summary1) and len(summary2) and len(summary3):
                part2_all.append(part2)
                summary2_all.append(summary2)
                summary.append(temp_data["summary"])

                # print(part3)
                part1_token = tokenizer1(part1, max_length=2000, truncation=True)
                part1_all.append(part1_token['input_ids'])
                # print(part1_all)
                # part2_token = tokenizer1(part2, max_length=4096, truncation=True, padding=True)
                part3_token = tokenizer1(part3, max_length=2000, truncation=True)
                part3_all.append(part3_token['input_ids'])

                summary3_token = tokenizer1(f_summary3, max_length=2000, truncation=True)
                summary3_all.append(summary3_token['input_ids'])

                bert_formatter = LawformerFormatter(800)
                # input_ids1, token_type_ids1, attention_mask1, cls_ids1 = bert_formatter.cls_process(part1_token)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = bert_formatter.cls_process(part3_token)

                input3_input_ids.append(input_ids3)
                input3_token_type_ids.append(token_type_ids3)
                input3_attention_mask.append(attention_mask3)
                input3_cls_ids.append(cls_ids3)
                input3_cls_len.append(cls_len3)

                # output.append(summary['input_ids'])

            # else:
            #     break

        else:
            # if len(part1) and len(part2) and len(part3):
                # part1_all.append(part1)
                part2_all.append(part2)
                # part3_all.append(part3)

                summary.append(temp_data["summary"])

                part1_token = tokenizer1(part1, max_length=2000, truncation=True)
                part1_all.append(part1_token['input_ids'])
                # part2_token = tokenizer1(part2, max_length=4096, truncation=True, padding=True)
                part3_token = tokenizer1(part3, max_length=2000, truncation=True)
                part3_all.append(part3_token['input_ids'])

                bert_formatter = LawformerFormatter(800)
                # input_ids1, token_type_ids1, attention_mask1, cls_ids1 = bert_formatter.cls_process(part1_token)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = bert_formatter.cls_process(part3_token)

                input3_input_ids.append(input_ids3)
                input3_token_type_ids.append(token_type_ids3)
                input3_attention_mask.append(attention_mask3)
                input3_cls_ids.append(cls_ids3)
                input3_cls_len.append(cls_len3)

            # else:
            #     break

    input3_input_ids = torch.LongTensor(input3_input_ids)
    input3_token_type_ids = torch.LongTensor(input3_token_type_ids)
    input3_attention_mask = torch.LongTensor(input3_attention_mask)
    input3_cls_ids = torch.LongTensor(input3_cls_ids)
    input3_cls_len = torch.LongTensor(input3_cls_len)

    summary_formatter = SummaryFormatter()
    part1_all_list = summary_formatter.summary_process(part1_all)
    part1_all = torch.LongTensor(part1_all_list)

    part3_all_list = summary_formatter.summary_process(part3_all)
    # print(part1_all_list)
    part3_all = torch.LongTensor(part3_all_list)

    # print(summary)
    summary_token = tokenizer1(summary, max_length=2000, truncation=True, padding=True, return_tensors='pt')
    summary = summary_token['input_ids']
    # print(summary)

    if mode != "test":
        summary3_all_list = summary_formatter.summary_process(summary3_all)
        # print(part1_all_list)
        summary3_all = torch.LongTensor(summary3_all_list)

        part2_token = tokenizer1(part2_all, max_length=2000, truncation=True, padding=True, return_tensors='pt')
        # summary1_token = tokenizer2(summary1_all, max_length=512, truncation=True, padding=True)
        summary2_token = tokenizer1(summary2_all, max_length=2000, truncation=True, padding=True, return_tensors='pt')
        # summary3_token = tokenizer2(summary3_all, max_length=512, truncation=True, padding=True)
        input2_input_ids = part2_token['input_ids']
        input2_token_type_ids = part2_token['token_type_ids']
        input2_attention_mask = part2_token['attention_mask']
        # output1 = summary1_token['input_ids']
        output2_input_ids = summary2_token['input_ids']
        # output2_attention_mask = summary2_token['attention_mask']
        # output3 = summary3_token['input_ids']
        return {
                'part1_all': part1_all,
                'input2_input_ids': input2_input_ids,
                'input2_token_type_ids': input2_token_type_ids,
                'input2_attention_mask': input2_attention_mask,
                'part3_all': part3_all,
                'input3_input_ids': input3_input_ids,
                'input3_token_type_ids': input3_token_type_ids,
                'input3_attention_mask': input3_attention_mask,
                'input3_cls_ids': input3_cls_ids,
                'input3_cls_len': input3_cls_len,
                'output2_input_ids': output2_input_ids,
                'output3': summary3_all,
                'summary': summary
                }
    else:
        part2_token = tokenizer1(part2_all, max_length=2000, truncation=True, padding=True, return_tensors='pt')
        input2 = part2_token['input_ids']
        return {
                'part1_all': part1_all,
                'input2': input2,
                'part3_all': part3_all,
                'input3_input_ids': input3_input_ids,
                'input3_token_type_ids': input3_token_type_ids,
                'input3_attention_mask': input3_attention_mask,
                'input3_cls_ids': input3_cls_ids,
                'input3_cls_len': input3_cls_len
                }

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

jieba.load_userdict('./data/user_dict.txt')
jieba.initialize()

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

# print(item['summary'])

# # part1
# part1_all = item['part1_all']

# # print(part1)
# part1_str = []
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# for batch in part1_all:
#     part1_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     part1_s = ["".join(str.replace(" ", "")) for str in part1_s]
#     part1_str.append(part1_s)
# # print(part1_str)

# user_dict = []
# with open('./data/user_dict.txt', encoding='utf-8') as f:
#     for l in f:
#         w = l.split()[0]
#         user_dict.append(w)  
# part1_s = []
# for batch in part1_str:
#     s = []
#     for i in batch:
#         if i in user_dict:
#             s.append(i)
#     part1_s.append(s)

# part1_summary = []
# for batch in part1_s:
#     summary = '原被告系'+batch[0]+'关系'
#     part1_summary.append(summary)
# print(part1_summary)
# # print(item['output1'])


# # bert encode layer------>lawformer
# # bert = BertModel.from_pretrained('bert-base-chinese')
# lawformer = AutoModel.from_pretrained('thunlp/Lawformer')
# input_ids = item['input3_input_ids']
# token_type_ids = item['input3_token_type_ids']
# attention_mask = item['input3_attention_mask']
# cls_ids = item['input3_cls_ids']
# cls_len = item['input3_cls_len']

# input_ids2 = item['input2_input_ids']
# token_type_ids2 = item['input2_token_type_ids']
# attention_mask2 = item['input2_attention_mask']

# print(input_ids.shape)
# print(token_type_ids.shape)
# print(attention_mask.shape)
# print(input_ids2.shape)
# print(token_type_ids2.shape)
# print(attention_mask2.shape)

# input_ids23 = torch.cat([input_ids,input_ids2],dim=1)
# token_type_ids23 = torch.cat([token_type_ids,token_type_ids2],dim=1)
# attention_mask23 = torch.cat([attention_mask,attention_mask2],dim=1)
# print(input_ids23.shape)
# print(token_type_ids23.shape)
# print(attention_mask23.shape)


# # print(cls_ids)

# outputs = lawformer(input_ids, token_type_ids, attention_mask)
# last_hidden = outputs[0]

# # print(cls_len)

# cls_inputs = []
# # cls_len = []
# for i, cls_id in enumerate(cls_ids):
#     # print(i)
#     # print(cls_id)
#     cls_tokens = []
#     # print(len(cls_id))//////300
#     # cls_len.append(len(cls_id))
#     # 截取cls_id中cls_len部分
#     # print(cls_len[i].item())
#     cls_id = cls_id[:cls_len[i].item()]
#     # print(cls_id)
#     for j in cls_id:
#         # print(last_hidden[i][j])
#         cls_tokens.append(last_hidden[i][j])
#     if len(cls_tokens) >= 300:  # max_cls_len
#         cls_tokens = cls_tokens[:300]
#     else:
#         for n in range(len(cls_tokens), 300):
#             cls_tokens.append(torch.LongTensor([0]*768))
#     # print(cls_tokens)
#     cls_tokens = torch.stack(cls_tokens, 0)
#     # print(cls_tokens)
#     cls_tokens = cls_tokens.unsqueeze(0)
#     # print(cls_tokens)
#     # print(cls_tokens.shape)
#     cls_inputs.append(cls_tokens)

# # print(cls_inputs)

# sen_inputs = cls_inputs[0]
# for i, sen in enumerate(cls_inputs):
#     if i == 0:
#         continue
#     else:
#         sen_inputs = torch.cat([sen_inputs, sen], dim=0)
# # print(sen_inputs)
# # print(sen_inputs.shape)
# # print(cls_len)

# # bi-lstm layer
# # cls_len_tensor = torch.tensor(cls_len)############################
# cls_len_tensor= cls_len
# # 使lstm传播屏蔽padding位的影响
# x_packed = rnn_utils.pack_padded_sequence(sen_inputs, cls_len_tensor, batch_first=True, enforce_sorted=False)
# lstm = nn.LSTM(768, 768, bidirectional=True, batch_first=True)
# fc = nn.Linear(768 * 2, 768)
# lstm.flatten_parameters()
# y_lstm, hidden = lstm(x_packed)# batch_size x T x input_size -> batch_size x T x (2*hidden_size)
# # print(cls_len_tensor)
# # print('lstm')
# # print(y_lstm)
# # print(y_lstm.shape)
# y_padded, length = rnn_utils.pad_packed_sequence(y_lstm, batch_first=True, total_length=300)#######
# lstm_output = fc(y_padded)  # batch_size x T x output_size

# # print(lstm_output)
# # print(lstm_output.shape)

# # multi head  attention layer
# w_q = nn.Linear(768, 768)
# w_k = nn.Linear(768, 768)
# w_v = nn.Linear(768, 768)
# fc = nn.Linear(768, 768)
# do = nn.Dropout(0.1)
# # 缩放
# scale = torch.sqrt(torch.FloatTensor([768 // 8]))
# batch_size = lstm_output.shape[0]
# Q = w_q(lstm_output)
# K = w_k(lstm_output)
# V = w_v(lstm_output)
# Q = Q.view(batch_size, -1, 8, 768 // 8).permute(0, 2, 1, 3)
# K = K.view(batch_size, -1, 8, 768 // 8).permute(0, 2, 1, 3)
# V = V.view(batch_size, -1, 8, 768 // 8).permute(0, 2, 1, 3)

# attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
# # 看attention的表示，修改padding的地方
# # mask为1的地方修改attention值
# # cls_len:tensor([31,24,23])   T->lstm_output.shape[1]  [3,8,31,31]
# mask_batch = []
# for i in range(batch_size):
#     mask_head = []
#     for n in range(8):#head
#         mask = np.ones((300, 300))
#         for j in range(cls_len[i].item()):
#             for k in range(cls_len[i].item()):
#                 mask[j][k] = 0
#         mask_head.append(mask)
#     mask_batch.append(mask_head)
# mask_batch = torch.LongTensor(np.array(mask_batch))
# # print(mask_batch.shape)
# attention = attention.masked_fill(mask_batch, -float('inf'))

# attention = do(torch.softmax(attention, dim=-1))
# attention_output = torch.matmul(attention, V)
# attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
# attention_output = attention_output.view(batch_size, -1, 8 * (768 // 8))
# attention_output = fc(attention_output)
# # print(attention_output)
# # print(attention_output.shape)

# # classifier layer
# fc = nn.Linear(768, 2)
# h = fc(attention_output)
# sent_scores = h.softmax(dim=2)
# out = sent_scores.argmax(dim=2)
# # print(sent_scores.shape)
# # 将sent_score中的[nan,nan]值修改为[0,1]
# # 以下用法与LawformerCLSEncoder中相同，重新构造sent——score
# sent_scores_list = []
# for batch in sent_scores:
#     sent_scores_list_batch = []
#     for score in batch:
#         # print(score)
#         if torch.isnan(score).any():
#             sent_scores_list_batch.append(torch.LongTensor([1,0]))
#         else:
#             sent_scores_list_batch.append(score)
#     sent_scores_list_batch = torch.stack(sent_scores_list_batch, 0)
#     sent_scores_list_batch = sent_scores_list_batch.unsqueeze(0)
#     # print(sent_scores_list_batch)
#     sent_scores_list.append(sent_scores_list_batch)
# sent_scores = sent_scores_list[0]
# for i, sen in enumerate(sent_scores_list):
#     if i == 0:
#         continue
#     else:
#         sent_scores = torch.cat([sent_scores, sen], dim=0)

# # print(sent_scores)
# # print(out)

# part3_all = item['part3_all']
# output3 = item['output3']

# # 进行原文抽取label标注

# def do_label( summary, document ):
#     y = [0] * len(document)
#     document_start = 0
#     for i in range(0, len(summary)):
#         if summary[i] in document[document_start:]:
#             _document_start = document[document_start:].index(summary[i])
#             y[document_start + _document_start] = 1
#             document_start = document_start +  _document_start + 1
#     return y

# part3_str = []
# # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# for batch in part3_all:
#     part3_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     part3_s = ["".join(str.replace(" ", "")) for str in part3_s]
#     part3_str.append(part3_s)

# output3_str = []
# # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# for batch in output3:
#     output3_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     output3_s = ["".join(str.replace(" ", "")) for str in output3_s]
#     output3_str.append(output3_s)  

# # print(part3_str)
# # print(output3_str)

# label_all = []
# for batch in range(len(output3_str)): #batch_size
#     label = do_label(output3_str[batch], part3_str[batch])
#     if len(label) >= sent_scores.shape[1]:
#         label = label[:sent_scores.shape[1]]
#     else:
#         for i in range(sent_scores.shape[1]-len(label)):
#             label.append(0)
#     # print(len(label))
#     label_all.append(label)
# # print(label_all)

# label_all = torch.LongTensor(np.array(label_all))

# sent_scores = sent_scores.transpose(1,2)
# # print(sent_scores.shape)
# # print(label_all.shape)

# criterion = torch.nn.CrossEntropyLoss()
# l1 = criterion(sent_scores, label_all)
# print(l1)






# # 抽取摘要
# summary = []
# for i, batch in enumerate(out):
#     out_batch = []
#     # print(batch)
#     for j, predict in enumerate(batch):
#         if predict.item() == 1:
#             out_batch.append(part3_str[i][j])
#     out_batch = ''.join(out_batch)
#     # print(out_batch)
#     summary.append(out_batch)
# # print(summary)

# summary_all = item['summary']
# summary_str = []
# # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# for batch in summary_all:
#     summary_s = tokenizer.batch_decode(batch, skip_special_tokens=True)
#     # summary_s = ["".join(str.replace(" ", "")) for str in summary_s]
#     summary_s = ''.join(summary_s)
#     summary_str.append(summary_s)
# # print(summary_str)

# # # train loss
# # tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# # summary = tokenizer1(summary, max_length=4096, truncation=True, padding=True)
# # print(summary['input_ids'])
# # print(item['output1'])


# # file = r"./data/vocab.txt"
 
# # # 第一种，直接打开文件读取行数（文件较小时）
# # count = 1
# # for count, line in enumerate(open(file, 'r', encoding='utf-8').readlines()):
# #     count += 1
# # print('行数:', count)#21128

# def fun( summary, document ):
#     y = [0] * len(document)
#     document_start = 0
#     for i in range(0, len(summary)):
#         if summary[i] in document[document_start:]:
#             _document_start = document[document_start:].index(summary[i])
#             y[document_start + _document_start] = 1
#             document_start = document_start +  _document_start + 1
#     return y


# # summary = ['a','b','c','d','e','f','g']
# # document = ['b','a','b','c','f','a','g','e','f','x','x','g','a']
#             #0   1   1   1   0   0   0   1   1   0   0   1   0

# document = ['综上所述', '，', '依照', '《', '中华人民共和国', '合同法', '》', '第六', '，', '《', '最高人民法院', '关于', '民事诉讼', '证据', '的', '若干', '规定', '》', '第二条', '，', '《', '中华人民共和国', '民事诉讼法', '》', '第一百四十四条', '规定', '，', '判决', '如下', '：', '驳回', '原告', '全贵良', '的', '全部', '诉讼请求', '。', '案件', '受理费', '人民币', '2400', '元', '，', '由', '原告', '全贵良', '负担', '。', '如', '不服', '本', '判决', '，', '可以', '在', '判决书', '送达', '之日起', '十五日', '内', '，', '向', '本院', '递交', '上诉状', '，', '并', '按', '对方', '当事人', '的', '人数', '提出', '副本', '，', '上诉', '于', '四川省', '遂宁市', '中级', '人民法院', '。']

# summary = ['依照', '《', '中华人民共和国', '合同法', '》', '第六', '，', '《', '最高人民法院', '关于', '民事诉讼', '证据', '的', '若干', '规定', '》', '第二条', '，', '《', '中华人民共和国', '民事诉讼法', '》', '第一百四十四条', '之', '规定', '，', '判决', '驳回', '原告', '全贵良', '的', '全部', '诉讼请求', '。'] 

# print(fun( summary, document ))

# summary1 = ['a','d','g']
# summary2 = ['b','e','h']
# summary3 = ['c','f','i']

# summary = []
# for i in range(len(summary1)):
#     s = summary1[i]+summary2[i]+summary3[i]
#     summary.append(s)
# print(summary)
# import sys,os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
# sys.path.append(BASE_DIR)

# from tools.accuracy_tool import get_rouge


# label = ['天天干家务烦死了','难受死了啊','中国']
# summary = ['这也完全不相干啊','真的难受死了啊','中华人民共和国']

# rouge = get_rouge(label, summary, 1)
# print(rouge)


# import json

# def get_data(num, data):
#     num1 = int(num * 0.8) + 1
#     num2 = int(num * 0.1)
#     num3 = num2

#     train_data = data[:num1]
#     valid_data = data[num1:num1+num2]
#     test_data = data[num1+num2:]

#     with open('./data/train.json', 'a+', encoding='utf-8') as f1:
#         for line in train_data:
#             json.dump(obj=line, fp=f1, ensure_ascii=False)
#             f1.write('\n')

#     with open('./data/valid.json', 'a+', encoding='utf-8') as f2:
#         for line in valid_data:
#             json.dump(obj=line, fp=f2, ensure_ascii=False)
#             f2.write('\n')

#     with open('./data/test.json', 'a+', encoding='utf-8') as f3:
#         for line in test_data:
#             json.dump(obj=line, fp=f3, ensure_ascii=False)
#             f3.write('\n')
    

# data = []
# num = 0
# with open('./data/data.json', encoding="utf-8") as f:
#     for line in f:
#         num = num + 1
#         data.append(json.loads(line))
# print(num)
# # get_data(num, data)

# num1 = 0
# num2 = 0
# num3 = 0
# with open('./data/train.json', encoding="utf-8") as f:
#     for line in f:
#         num1 = num1 + 1
# with open('./data/valid.json', encoding="utf-8") as f:
#     for line in f:
#         num2 = num2 + 1
# with open('./data/test.json', encoding="utf-8") as f:
#     for line in f:
#         num3 = num3 + 1
# print(num1)
# print(num2)
# print(num3)


# a = [1,2,3,4,5,6]

# print(a[1:-2])

# python3 train.py --config config/LCASE.config --gpu 0,1

# def a(x):
#     if x == 1:
#         return 1
#     return 0

# print(a(0))

import torch


a = torch.LongTensor([[101,134]])
print(a[:,-1].unsqueeze(1))
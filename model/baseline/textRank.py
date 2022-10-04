import numpy as np
import pandas as pd
import re
import json
import jieba.analyse
from rouge import Rouge
import heapq

import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目�?
sys.path.append(BASE_DIR)
sys.setrecursionlimit(2000)

input_path = "./data/test.json"
output_path = "./output/textRank/result.json"

def load_user_dict(filename):
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict

def sentence_split(sentence):
    result = jieba.lcut(sentence)
    return result

def document_split(document):
    start = 0
    result = []
    groups = re.finditer('。|，|；', document)

    for i in groups:
        end = i.span()[1]
        result.append(document[start:end])
        start = end
    # last one
    result.append(document[start:])

    return result

def get_pro_document(text):
    document = ""
    for i, _ in enumerate(text):
        sent_text = text[i]["sentence"]
        document = document+sent_text
    return document

def textRank(document):
    words = []
    scores = []
    pro_document = []
    sentence_score = []
    pre_summary = []
    for x, w in jieba.analyse.textrank(document, topK=100, withWeight=True, allowPOS=('ns', 'n', 'vn', 'n')):
        # print(x, w)
        words.append(x)
        scores.append(w)
    document = document_split(document)
    for sentence in document:
        result = sentence_split(sentence)
        pro_document.append(result)
    for sentence in pro_document:
        score = 0
        for i,x in enumerate(words):
            if x in sentence:
                score = score+scores[i]
        sentence_score.append(score)
    max_list = list(map(sentence_score.index, heapq.nlargest(12, sentence_score)))
    max_list.sort()
    for i in max_list:
        pre_summary.append(document[i])
    pre_summary = ''.join(pre_summary)
    # print(pre_summary)
    # print('\n')
    return pre_summary

# if __name__ == "__main__":
#     jieba.load_userdict('./data/user_dict.txt')
#     jieba.initialize()
#     with open(output_path, 'a', encoding='utf8') as fw:
#         with open(input_path, 'r', encoding="utf8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 id = data.get('id')
#                 text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
#                 summary = data.get('summary')
#                 document = get_pro_document(text)
#                 pre_summary = textRank(document)  # your model predict
#                 result = dict(
#                     id=id,
#                     summary=summary,
#                     pre_summary=pre_summary
#                 )
#                 fw.write(json.dumps(result, ensure_ascii=False) + '\n')

rouge = Rouge()
def get_rouge(pre_summary, summary):
    """计算rouge-1、rouge-2、rouge-l
    """
    # pre_summary = [" ".join(jieba.cut(pre.replace(" ", ""))) for pre in pre_summary]
    # summary = [" ".join(jieba.cut(lab.replace(" ", ""))) for lab in summary]

    pre_summary = [" ".join(pre.replace(" ", "")) for pre in pre_summary]
    summary = [" ".join(lab.replace(" ", "")) for lab in summary]
    try:
        scores = rouge.get_scores(hyps=pre_summary, refs=summary)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

if __name__ == "__main__":
    with open(output_path, 'r', encoding='utf8') as f:
        rouge_1 = 0
        rouge_2 = 0
        rouge_l = 0
        num = 0
        for line in f:
            data = json.loads(line)
            pre_summary = [data.get('pre_summary')]
            summary = [data.get('summary')]
            rouge_score = get_rouge(pre_summary,summary)  # your model predict
            rouge_1 = rouge_1 + rouge_score['rouge-1']
            rouge_2 = rouge_2 + rouge_score['rouge-2']
            rouge_l = rouge_l + rouge_score['rouge-l']
            num = num+1
        rouge_1 = rouge_1/num
        rouge_2 = rouge_2/num
        rouge_l = rouge_l/num
        print('rouge-1:'+str(rouge_1))
        print('rouge-2:'+str(rouge_2))
        print('rouge-l:'+str(rouge_l))

# rouge-1:0.5440446261843841
# rouge-2:0.2855371435184395
# rouge-l:0.4006926703543477
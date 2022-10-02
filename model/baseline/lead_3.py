# coding=utf-8

import json
import re
from rouge import Rouge

input_path = "./data/test.json"
output_path = "./output/lead_3/result.json"


# def get_summary(text):
#     for i, _ in enumerate(text):
#         sent_text = text[i]["sentence"]
#         if re.search(r"诉讼请求", sent_text):
#             text0 = text[i]["sentence"]
#             text1 = text[i + 1]["sentence"]
#             text2 = text[i + 2]["sentence"]
#             break
#         else:
#             text0 = text[11]["sentence"]
#             text1 = text[12]["sentence"]
#             text2 = text[13]["sentence"]
#     result = text0 + text1 + text2
#     return result


# if __name__ == "__main__":
#     with open(output_path, 'a', encoding='utf8') as fw:
#         with open(input_path, 'r', encoding="utf8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 id = data.get('id')
#                 text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
#                 summary = data.get('summary')
#                 pre_summary = get_summary(text)  # your model predict
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

# rouge-1:0.463053584657249
# rouge-2:0.22843210677971368
# rouge-l:0.3531930575985233    
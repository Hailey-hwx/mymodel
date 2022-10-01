# import sys,os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目??
# sys.path.append(BASE_DIR)


import torch
from transformers import BertTokenizer
from .DucomentSegment import DocumentSegment
from .SummarySegment import SummarySegment
from .SummaryFormatter import SummaryFormatter
from formatter.LawformerFormatter import LawformerFormatter
from .Basic import BasicFormatter


class ModelFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.max_lawformer_len = config.getint("data", "max_lawformer_len")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
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
        summary_all = []

        for temp_data in data:
            idx.append(temp_data["id"])
            document = []
            for text in temp_data["text"]:
                document.append(text["sentence"])
            # 
            document_segment = DocumentSegment()
            part1, part2, part3 = document_segment.get_part(document)
            # 
            part2 = ''.join(part2)

            if mode != "test":
                summary_segment = SummarySegment()
                summary1, summary2, summary3 = summary_segment.get_part(temp_data["summary"])
                summary1 = ''.join(summary1)
                summary2 = ''.join(summary2)
                summary3 = ''.join(summary3)
                summary3 = document_segment.sentence_split(summary3)
                f_summary3 = []
                for s in summary3:
                    if s != '':
                        f_summary3.append(s)

                # 删除空数据
                # if len(part1) and len(part2) and len(part3) and len(summary1) and len(summary2) and len(summary3):
                part2_all.append(part2)
                summary2_all.append(summary2)
                summary_all.append(temp_data["summary"])

                part1_token = self.tokenizer(part1, max_length=self.max_lawformer_len, truncation=True)
                part1_all.append(part1_token['input_ids'])

                part3_token = self.tokenizer(part3, max_length=self.max_lawformer_len, truncation=True)
                part3_all.append(part3_token['input_ids'])

                summary3_token = self.tokenizer(f_summary3, max_length=self.max_lawformer_len, truncation=True)
                summary3_all.append(summary3_token['input_ids'])

                lawformer_formatter = LawformerFormatter(config)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = lawformer_formatter.cls_process(part3_token)

                input3_input_ids.append(input_ids3)
                input3_token_type_ids.append(token_type_ids3)
                input3_attention_mask.append(attention_mask3)
                input3_cls_ids.append(cls_ids3)
                input3_cls_len.append(cls_len3)    

                # else:
                #     break

            else:
                # if len(part1) and len(part2) and len(part3):
                part2_all.append(part2)

                summary_all.append(temp_data["summary"])

                part1_token = self.tokenizer(part1, max_length=self.max_lawformer_len, truncation=True)
                part1_all.append(part1_token['input_ids'])

                part3_token = self.tokenizer(part3, max_length=self.max_lawformer_len, truncation=True)
                part3_all.append(part3_token['input_ids'])

                lawformer_formatter = LawformerFormatter(config)
                input_ids3, token_type_ids3, attention_mask3, cls_ids3, cls_len3 = lawformer_formatter.cls_process(part3_token)

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
        part3_all = torch.LongTensor(part3_all_list)

        summary_token = self.tokenizer(summary_all, max_length=self.max_lawformer_len, truncation=True, padding=True, return_tensors='pt')
        summary = summary_token['input_ids']

        if mode != "test":
            summary3_all_list = summary_formatter.summary_process(summary3_all)
            summary3_all = torch.LongTensor(summary3_all_list)

            part2_token = self.tokenizer(part2_all, max_length=self.max_lawformer_len, truncation=True, padding=True,
                                          return_tensors='pt')
            summary2_token = self.tokenizer(summary2_all, max_length=self.max_lawformer_len, truncation=True, padding=True, return_tensors='pt')
            input2_input_ids = part2_token['input_ids']
            input2_attention_mask = part2_token['attention_mask']
            output2_input_ids = summary2_token['input_ids']
            return {
                    'part1_all': part1_all,
                    'input2_input_ids': input2_input_ids,
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
            part2_token = self.tokenizer(part2_all, max_length=self.max_lawformer_len, truncation=True, padding=True,
                                          return_tensors='pt')
            input2_input_ids = part2_token['input_ids']
            input2_attention_mask = part2_token['attention_mask']       
            return {
                    'part1_all': part1_all,
                    'input2_input_ids': input2_input_ids,
                    'input2_attention_mask': input2_attention_mask,
                    'part3_all': part3_all,
                    'input3_input_ids': input3_input_ids,
                    'input3_token_type_ids': input3_token_type_ids,
                    'input3_attention_mask': input3_attention_mask,
                    'input3_cls_ids': input3_cls_ids,
                    'input3_cls_len': input3_cls_len,
                    'summary': summary
                }

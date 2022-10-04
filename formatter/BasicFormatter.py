# import sys,os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# sys.path.append(BASE_DIR)


import torch
from transformers import BertTokenizer
from .DucomentSegment import DocumentSegment
from .SummarySegment import SummarySegment
from .SummaryFormatter import SummaryFormatter
from formatter.LawformerFormatter import LawformerFormatter
from .Basic import BasicFormatter


class BasicFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.max_bert_len = config.getint("data", "max_bert_len")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        document_all = []
        input_input_ids = []
        input_token_type_ids = []
        input_attention_mask = []
        input_cls_ids = []
        input_cls_len = []
        idx = []
        summary_all = []

        for temp_data in data:
            idx.append(temp_data["id"])
            document = []
            for text in temp_data["text"]:
                document.append(text["sentence"])
            document = ''.join(document)
            document_segment = DocumentSegment()
            document = document_segment.sentence_split(document)
            f_document = []
            for s in document:
                if s != '':
                    f_document.append(s)

            if mode != "test":
                summary = document_segment.sentence_split(temp_data["summary"])
                f_summary = []
                for s in summary:
                    if s != '':
                        f_summary.append(s)

                _summary.append(temp_data["summary"])

                document_token = tokenizer(f_document, max_length=512, truncation=True)
                document_all.append(document_token['input_ids'])

                summary_token = tokenizer(f_summary, max_length=512, truncation=True)
                summary_all.append(summary_token['input_ids'])

                bert_formatter = LawformerFormatter(512)
                input_ids, token_type_ids, attention_mask, cls_ids, cls_len = bert_formatter.cls_process(document_token)

                input_input_ids.append(input_ids)
                input_token_type_ids.append(token_type_ids)
                input_attention_mask.append(attention_mask)
                input_cls_ids.append(cls_ids)
                input_cls_len.append(cls_len)    

            else:
                document_token = tokenizer(f_document, max_length=512, truncation=True)
                document_all.append(document_token['input_ids'])

                _summary.append(temp_data["summary"])

                bert_formatter = LawformerFormatter(512)
                input_ids, token_type_ids, attention_mask, cls_ids, cls_len = bert_formatter.cls_process(document_token)

                input_input_ids.append(input_ids)
                input_token_type_ids.append(token_type_ids)
                input_attention_mask.append(attention_mask)
                input_cls_ids.append(cls_ids)
                input_cls_len.append(cls_len)

        input_input_ids = torch.LongTensor(input_input_ids)
        input_token_type_ids = torch.LongTensor(input_token_type_ids)
        input_attention_mask = torch.LongTensor(input_attention_mask)
        input_cls_ids = torch.LongTensor(input_cls_ids)
        input_cls_len = torch.LongTensor(input_cls_len)

        summary_formatter = SummaryFormatter()

        document_all_list = summary_formatter.summary_process(document_all)
        document_all = torch.LongTensor(document_all_list)

        _summary_token = tokenizer(_summary, max_length=512, truncation=True, padding=True, return_tensors='pt')
        _summary = _summary_token['input_ids']

        if mode != "test":
            summary_all_list = summary_formatter.summary_process(summary_all)
            summary_all = torch.LongTensor(summary_all_list)
            return {
                    'document_all': document_all,
                    'input_input_ids': input_input_ids,
                    'input_token_type_ids': input_token_type_ids,
                    'input_attention_mask': input_attention_mask,
                    'input_cls_ids': input_cls_ids,
                    'input_cls_len': input_cls_len,
                    'summary_all': summary_all,
                    '_summary': _summary
                    }
        else:
            return {
                    'document_all': document_all,
                    'input_input_ids': input_input_ids,
                    'input_token_type_ids': input_token_type_ids,
                    'input_attention_mask': input_attention_mask,
                    'input_cls_ids': input_cls_ids,
                    'input_cls_len': input_cls_len,
                    '_summary': _summary
                    }
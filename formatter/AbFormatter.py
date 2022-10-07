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


class AbFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.max_bert_len = config.getint("data", "max_bert_len")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        document_all = []
        idx = []
        summary_all = []

        for temp_data in data:
            idx.append(temp_data["id"])
            document = []
            for text in temp_data["text"]:
                document.append(text["sentence"])
            document = ''.join(document)

            summary = temp_data["summary"]
            document_all.append(document)
            summary_all.append(summary)
        
        document_token = self.tokenizer(document_all, max_length=3000, truncation=True, padding=True, return_tensors='pt')
        document_input_ids = document_token['input_ids']

        summary_token = self.tokenizer(summary_all, max_length=3000, truncation=True, padding=True, return_tensors='pt')
        summary_input_ids = summary_token['input_ids']
        
        return {
                'document_input_ids': document_input_ids,
                'summary_input_ids': summary_input_ids
                }
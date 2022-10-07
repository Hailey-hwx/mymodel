import torch
import torch.nn as nn
from transformers import BertModel


class BertCLSEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LawformerCLSEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.max_cls_len = config.getint("data", "max_cls_len")
        self.hidden_size = config.getint("model", "hidden_size")

    def forward(self, input_ids, token_type_ids, attention_mask, cls_ids, cls_len):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        last_hidden = outputs[0]

        cls_inputs = []

        for i, cls_id in enumerate(cls_ids):
            cls_tokens = []
            cls_id = cls_id[:cls_len[i].item()]
            for j in cls_id:
                cls_tokens.append(last_hidden[i][j])
            if len(cls_tokens) >= self.max_cls_len:
                cls_tokens = cls_tokens[:self.max_cls_len]
            else:
                for n in range(len(cls_tokens), self.max_cls_len):
                    cls_tokens.append((torch.LongTensor([0] * self.hidden_size).cuda()))
            cls_tokens = torch.stack(cls_tokens, 0)
            cls_tokens = cls_tokens.unsqueeze(0)
            cls_inputs.append(cls_tokens)

        sen_inputs = cls_inputs[0]
        for i, sen in enumerate(cls_inputs):
            if i == 0:
                continue
            else:
                sen_inputs = torch.cat([sen_inputs, sen], dim=0)

        return sen_inputs

class Bert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Bert, self).__init__()
        self.bert = BertCLSEncoder(config, gpu_list, *args, **params)
        self.classifier = Classifier(config, gpu_list, *args, **params)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        if config.getboolean("distributed", "use"):
            self.bert = nn.parallel.DistributedDataParallel(self.bert, device_ids=device)
            self.classifier = nn.parallel.DistributedDataParallel(self.classifier, device_ids=device)
        else:
            self.bert = nn.DataParallel(self.bert, device_ids=device)
            self.classifier = nn.DataParallel(self.classifier, device_ids=device)

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

    def forward(self, data, config, gpu_list, mode):
        x = data
        input_ids = x['input_input_ids']
        token_type_ids = x['input_token_type_ids']
        attention_mask = x['input_attention_mask']
        cls_ids = x['input_cls_ids']
        cls_len = x['input_cls_len']

        sen_inputs = self.bert(input_ids, token_type_ids, attention_mask, cls_ids, cls_len)
        sent_scores = self.classifier(sen_inputs)
        sent_out = sent_scores.argmax(dim=2)

        document_all = x['document_all']
        document_str = []
        for batch in document_all: 
            document_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            document_s = ["".join(str.replace(" ", "")) for str in document_s]
            document_str.append(document_s)
        # print(document_str)
        document_summary = []
        for i, batch in enumerate(sent_out):
            out_batch = []
            for j, predict in enumerate(batch):
                if predict.item() == 1:
                    out_batch.append(document_str[i][j])
            out_batch = ''.join(out_batch)
            document_summary.append(out_batch)#predict summary

        _summary = x['_summary']
        summary_str = []
        for batch in _summary:
            summary_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            summary_s = ''.join(summary_s)
            summary_str.append(summary_s)
        print(summary_str)

        if mode != "test":
            summary_all = x['summary_all']
            label = get_extractive_label(summary_all, document_str, sent_scores)
            
            sent_scores = sent_scores.transpose(1,2)
            loss = self.criterion(sent_scores, label)

            print(loss)

            # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
            return {"loss": loss, "summary": summary_str, "pre_summary": document_summary}


        # acc_result = self.accuracy_function(pre_summary, summary_str, config)#
        return {"summary": summary_str, "pre_summary": document_summary}


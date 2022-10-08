import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
print(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel, BertTokenizer
# from model.decoder.TransDecoder import TransMask, TextEmbedding, PositionalEnconding, Generator


class Seq2seq(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Seq2seq, self).__init__()
        self.vocab = config.getint("model", "vocab")
        self.d_model = config.getint("model", "hidden_size")
        self.embedding = nn.Embedding(self.vocab, self.d_model)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.lstm = nn.LSTM(self.d_model, self.d_model, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.d_model * 2, self.d_model)

        self.attn = nn.Linear((self.d_model * 2) + self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, 1, bias=False)
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.d_lstm = nn.LSTM(self.d_model*2+self.d_model, self.d_model, batch_first=True)
        self.d_fc = nn.Linear(self.d_model * 2+self.d_model+self.d_model, self.vocab)

        self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        if config.getboolean("distributed", "use"):
            self.embedding = nn.parallel.DistributedDataParallel(self.embedding, device_ids=device)
            self.dropout = nn.parallel.DistributedDataParallel(self.dropout, device_ids=device)
            self.lstm = nn.parallel.DistributedDataParallel(self.lstm, device_ids=device)
            self.fc = nn.parallel.DistributedDataParallel(self.fc, device_ids=device)
            self.attn = nn.parallel.DistributedDataParallel(self.attn, device_ids=device)
            self.v = nn.parallel.DistributedDataParallel(self.v, device_ids=device)
            self.d_lstm = nn.parallel.DistributedDataParallel(self.d_lstm, device_ids=device)
            self.d_fc = nn.parallel.DistributedDataParallel(self.d_fc, device_ids=device)
        else:
            self.mask = nn.DataParallel(self.mask, device_ids=device)
            self.embedding = nn.DataParallel(self.embedding, device_ids=device)
            self.dropout = nn.DataParallel(self.dropout, device_ids=device)
            self.lstm = nn.DataParallel(self.lstm, device_ids=device)
            self.fc = nn.DataParallel(self.fc, device_ids=device)
            self.attn = nn.DataParallel(self.attn, device_ids=device)
            self.v = nn.DataParallel(self.v, device_ids=device)
            self.d_lstm = nn.DataParallel(self.d_lstm, device_ids=device)
            self.d_fc = nn.DataParallel(self.d_fc, device_ids=device)

    def forward(self, data, config, gpu_list, mode):
        x = data

        document_input_ids = x['document_input_ids']
        summary_input_ids = x['summary_input_ids']

        _summary = summary_input_ids
        summary_str = []
        for batch in _summary:
            summary_s = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            summary_s = ''.join(summary_s)
            summary_str.append(summary_s)
        # print(summary_str)

        # encoder
        doc_len = []
        for batch in document_input_ids:
            len = 0
            for i in batch:
                if i != 0:
                    len=len+1
            doc_len.append(len)
        doc_len = torch.LongTensor(doc_len)

        embedded = self.dropout(self.embedding(document_input_ids))
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, doc_len, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        packed_outputs, hidden = self.lstm(packed_embedded) 
        outputs, length = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)#, , total_length=3000
        hidden = torch.tanh(self.fc(torch.cat((hidden[0][:, -2, :], hidden[0][:, -1, :]), dim=1)))

        # attention
        encoder_outputs = outputs
        batch_size = encoder_outputs.shape[0]
        doc_len = encoder_outputs.shape[1]
        a_hidden = hidden.unsqueeze(1).repeat(1, doc_len, 1)
        energy = torch.tanh(self.attn(torch.cat((a_hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        mask = (document_input_ids != 0)
        attention = attention.masked_fill(mask == 0, -1e10)
        a = attention.softmax(dim=1)

        # decoder
        sum_len = summary_input_ids.shape[1]
        decoder_outputs = torch.zeros(sum_len, batch_size, self.vocab)###cuda
        input = summary_input_ids[:,0]  #[101,101]
        for t in range(1,sum_len):
            input = input.unsqueeze(1)
            sum_embedded = self.dropout(self.embedding(input))  #[batchsize, 1, 768]
            weighted = torch.bmm(a.unsqueeze(1), encoder_outputs)   #[batchsize, 1, 768*2]
            sum_input = torch.cat((sum_embedded, weighted), dim=2)  #[batchsize, 1, 768*3]
            d_output, d_hidden = self.d_lstm(sum_input)
            sum_embedded = sum_embedded.squeeze(1) 
            d_output = d_output.squeeze(1)
            weighted = weighted.squeeze(1)
            prediction = self.d_fc(torch.cat((d_output, weighted, sum_embedded), dim = 1))
            decoder_outputs[t] = prediction
            pre_word = prediction.argmax(1)
            if mode == 'train':
                input = summary_input_ids[:,t]
            else:
                input = pre_word

        decoder_outputs = decoder_outputs.permute(1,2,0)  #[batchsize, vocab, len]
        loss = self.criterion(decoder_outputs, summary_input_ids)
        # print(loss)

        decoder_prediction = decoder_outputs.argmax(1)
        summary_temp = self.tokenizer.batch_decode(decoder_prediction, skip_special_tokens=True)
        pre_summary = ["".join(str.replace(" ", "")) for str in summary_temp]

        if mode == 'train':
            # print('train')
            return {"loss": loss, "summary": summary_str, "pre_summary": pre_summary}

        return {"summary": summary_str, "pre_summary": pre_summary}
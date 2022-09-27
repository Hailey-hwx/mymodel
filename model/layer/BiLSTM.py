import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class BiLSTM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BiLSTM, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")
        self.input_size = self.hidden_size
        self.output_size = self.hidden_size
        self.dropout = config.getfloat("model", "dropout")
        self.max_cls_len = config.getint("data", "max_cls_len")
        # lstm层数，这里默认设置为1，，，num_layer, 如果过拟合设置dropout
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, sen_inputs, cls_len):
        cls_len_tensor = cls_len

        cls_len_tensor = cls_len_tensor.cpu()

        # 使lstm传播屏蔽padding位的影响
        x_packed = rnn_utils.pack_padded_sequence(sen_inputs, cls_len_tensor, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        y_lstm, hidden = self.lstm(x_packed)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        y_padded, length = rnn_utils.pad_packed_sequence(y_lstm, batch_first=True, total_length=self.max_cls_len)
        lstm_output = self.fc(y_padded)  # batch_size x T x output_size 

        return lstm_output

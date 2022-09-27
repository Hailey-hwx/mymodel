
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Classifier, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, attention_output):
        h = self.fc(attention_output)
        sent_scores = h.softmax(dim=2)
        # out = sent_scores.argmax(dim=2)
        # out = out.numpy().tolist()

        return sent_scores

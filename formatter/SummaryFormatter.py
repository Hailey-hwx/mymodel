import torch


class SummaryFormatter:
    def __init__(self):
        super().__init__()

    # ������ɾ���token
    def summary_process(self, inputs):
        max_part_len = 0 # �ִ�������
        max_fc_len = 0 # ÿ���ִ���󳤶�
        for batch in inputs:
            max_part_len = max(max_part_len, len(batch))
            # print(batch)
            for s in batch:
                max_fc_len = max(max_fc_len, len(s))
        # print(max_part_len)
        # print(max_fc_len)

        for i, batch in enumerate(inputs):
            if len(batch) < max_part_len:
                for n in range(max_part_len-len(batch)):
                    inputs[i].append([0]*max_fc_len)
            for j,s in enumerate(batch):
                if len(s) < max_fc_len:
                    for n in range(max_fc_len-len(s)):
                        inputs[i][j].append(0)

        # inputs = torch.LongTensor()

        return inputs

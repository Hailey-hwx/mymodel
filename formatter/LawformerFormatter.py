import torch


class LawformerFormatter:
    def __init__(self, config):
        super().__init__()
        # self.max_len = config.getint("data", "max_len")
        # self.max_cls_len = config.getint("data", "max_cls_len")
        self.max_len = config
        self.max_cls_len = 200

    # 重新组成句子token
    def cls_process(self, inputs):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        cls_ids = []
        # cls_len = []

        # token处理
        for input_id in inputs['input_ids']:
            input_ids = input_ids + input_id
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = input_ids + [0] * (self.max_len - len(input_ids))

        for mask in inputs['attention_mask']:
            attention_mask = attention_mask + mask
        if len(attention_mask) >= self.max_len:  # max_len
            attention_mask = attention_mask[:self.max_len]
        else:
            attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))

        for i in range(len(inputs['token_type_ids'])):
            if i % 2:
                inputs['token_type_ids'][i] = [1] * len(inputs['token_type_ids'][i])
        for token_type_id in inputs['token_type_ids']:
            token_type_ids = token_type_ids + token_type_id

        if len(token_type_ids) >= self.max_len:  
            token_type_ids = token_type_ids[:self.max_len]
        else:
            token_type_ids = token_type_ids + [0] * (self.max_len - len(token_type_ids))

        for i in range(len(input_ids)):
            if input_ids[i] == 101:
                cls_ids.append(i)

        # cls_len = len(cls_ids)
        
        if len(cls_ids) >= self.max_cls_len:  
            cls_len = self.max_cls_len
            cls_ids = cls_ids[:self.max_cls_len]
        else:
            cls_len = len(cls_ids)
            cls_ids = cls_ids + [1001] * (self.max_cls_len - len(cls_ids))

        # input_ids = torch.LongTensor([input_ids])
        # token_type_ids = torch.LongTensor([token_type_ids])
        # attention_mask = torch.LongTensor([attention_mask])

        return input_ids, token_type_ids, attention_mask, cls_ids, cls_len

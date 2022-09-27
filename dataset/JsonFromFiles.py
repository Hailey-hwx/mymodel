
import json
import os
from torch.utils.data import Dataset


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        # config为读取的配置文件  mode为读取器的格式，包括train、valid、test
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")

        self.data = []
        for filename in filename_list:
            f = open(os.path.join(self.data_path, filename), "r", encoding="utf8")
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

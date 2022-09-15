import torch
import subprocess
import os
import random

class PredictNextDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_file_path, num_split_line=10000, cache_dir="predict_next_dataset_cache", separator="\t", prefix= "", max_ref=5):
        super().__init__()
        
        self.num_split_line = num_split_line
        self.cache_dir_path = cache_dir
        self.separator = separator
        self.prefix = prefix
        self.len = 0
        # initialize cache
        if not os.path.exists(self.cache_dir_path):
            os.mkdir(self.cache_dir_path)

        # load lines
        buff = []
        for path in list_of_file_path:
            with open(path) as f:
                buff += f.read().split("\n")
        lines = []
        for i in range(len(buff) - max_ref):
            num_ref = random.randint(2,max_ref)
            s = "".join(buff[i-num_ref:i-1]).replace(self.separator, "") + self.separator + buff[i].replace(self.separator, "")
            lines.append(s)
        self.len = len(lines)

        # save to cache
        c = 0
        for i in range(0, len(lines)-1, num_split_line):
            l = lines[i:i+num_split_line]
            with open(os.path.join(self.cache_dir_path, f"{c}.txt"), mode="w") as f:
                f.write("\n".join(l))
            c += 1

    def __getitem__(self, idx):
        fid = idx // self.num_split_line
        with open(os.path.join(self.cache_dir_path, f"{fid}.txt")) as f:
            src, tgt = f.read().split("\n")[idx % self.num_split_line].split("\t")
            return src, tgt


    def __len__(self):
        return self.len

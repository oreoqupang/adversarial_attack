import torch
from torch.utils.data import Dataset
import numpy as np

class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, padding_limit, first_n_byte=1000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.padding_limit = padding_limit
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                tmp = np.array([i+1 for i in f.read()[:self.first_n_byte]])
                origin_len = tmp.size
                padding_len = min(origin_len + self.padding_limit, self.first_n_byte) - origin_len
                
                tmp = np.concatenate((tmp, np.random.randint(1, 257, padding_len), np.zeros(self.first_n_byte-origin_len-padding_len)), axis=0 )

        except:
            with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
                tmp = np.array([i+1 for i in f.read()[:self.first_n_byte]])
                origin_len = tmp.size
                padding_len = min(origin_len + self.padding_limit, self.first_n_byte) - origin_len
                
                tmp = np.concatenate((tmp, np.random.randint(1, 257, padding_len), np.zeros(self.first_n_byte-origin_len-padding_len)), axis=0 )

        return torch.Tensor(tmp).long(), torch.Tensor([self.label_list[idx]]), torch.Tensor([origin_len]).long()
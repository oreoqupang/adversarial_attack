import torch
from torch.utils.data import Dataset
import numpy as np

class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=1000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                origin_len = len(tmp)
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))

        except:
            with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                origin_len = len(tmp)
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))

        return torch.Tensor(tmp), torch.Tensor([self.label_list[idx]]), torch.Tensor([origin_len])
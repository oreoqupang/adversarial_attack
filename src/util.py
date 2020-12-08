import torch
import json
from torch.utils.data import Dataset
import numpy as np
from argparse import ArgumentParser


def read_config(config_filepath):
    with open(f'resource/config/{config_filepath}') as f:
        config_dict = json.load(f)

    parser = ArgumentParser()
    for k, v in config_dict.items():
        parser.add_argument(f'--{k}', type=eval(v['type']), default=v['default'])

    config, _ = parser.parse_known_args()
    return config


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
        with open(self.data_path+self.fp_list[idx],'rb') as f:
            origin = np.array([i+1 for i in f.read()[:self.first_n_byte]])
            origin_len = origin.size
            padding_len = min(origin_len + self.padding_limit, self.first_n_byte) - origin_len

            tmp = np.concatenate((origin, np.zeros(self.first_n_byte-origin_len)), axis=0)     
            adv = np.concatenate((origin, np.random.randint(1, 257, padding_len), np.zeros(self.first_n_byte-origin_len-padding_len)), axis=0 )

        return torch.Tensor(adv).long(), torch.Tensor(tmp).long(), torch.Tensor([self.label_list[idx]]), torch.Tensor([origin_len]).long()

class ExeDataset_FGM(Dataset):
    def __init__(self, fp_list, data_path, label_list, padding_len, first_n_byte=1000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.padding_len = padding_len
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        with open(self.data_path+self.fp_list[idx],'rb') as f:
            origin = np.array([i+1 for i in f.read()])
            origin_len = origin.size

            tmp = np.concatenate((origin, np.zeros(self.first_n_byte-origin_len)), axis=0)
            adv = np.concatenate((origin, np.random.randint(1, 257, self.padding_len), np.zeros(self.first_n_byte-origin_len-self.padding_len)), axis=0 )

        return torch.Tensor(adv).long(), torch.Tensor(tmp).long(), torch.Tensor([self.label_list[idx]]), torch.Tensor([origin_len]).long()
import torch
import json
import numpy as np
import struct
import os
import lief
import random
import pandas as pd
from torch.utils.data import Dataset
from argparse import ArgumentParser

def insert_SLACK(target_bin, raw_bytes, ins_size):
    parsed = lief.parse(target_bin)

    optional_header_offset =  parsed.dos_header.addressof_new_exeheader + 0x18
    first_section_header = optional_header_offset + parsed.header.sizeof_optional_header
    
    ins_offset = np.inf
    ins_size += parsed.optional_header.file_alignment - ins_size%parsed.optional_header.file_alignment
    for i in range(parsed.header.numberof_sections):
        idx = first_section_header+i*0x28+0x14
        pointer_raw_byte = struct.unpack("<I", raw_bytes[idx:idx+4].astype(np.uint8).tobytes())[0]
        if ins_offset > pointer_raw_byte:
            ins_offset = pointer_raw_byte

    # modify each Section's pointer_raw_byte 
    for i in range(parsed.header.numberof_sections):
        idx = first_section_header+i*0x28+0x14
        pointer_raw_byte = struct.unpack("<I", raw_bytes[idx:idx+4].astype(np.uint8).tobytes())[0]
        if pointer_raw_byte >= ins_offset:
            raw_bytes[idx:idx+4] = np.frombuffer(struct.pack("<I", pointer_raw_byte+ins_size), dtype='uint8')
        
    # modify SizeOfImage
    raw_bytes[optional_header_offset+0x38:optional_header_offset+0x3c] = np.frombuffer(struct.pack("<I", parsed.optional_header.sizeof_image+ins_size), dtype='uint8')

    return np.insert(raw_bytes, ins_offset, np.random.randint(0, 256, ins_size)), ins_offset

def find_SLACK(target_bin):
    try:
        parsed = lief.parse(target_bin)
        file_len = os.stat(target_bin).st_size 
    except Exception as e:
        print("errrrrrr", e)
        return torch.empty(0)

    target_idxs = []
    for section in parsed.sections:
        if section.size > section.virtual_size:
            end = min(file_len, section.offset+section.size)
            target_idxs.extend(list(range(section.offset+section.virtual_size, end)))

    return torch.Tensor(target_idxs).long()

def read_config(config_filepath):
    with open(f'resource/config/{config_filepath}') as f:
        config_dict = json.load(f)

    parser = ArgumentParser()
    for k, v in config_dict.items():
        parser.add_argument(f'--{k}', type=eval(v['type']), default=v['default'])

    config, _ = parser.parse_known_args()
    return config

class PEDataset(Dataset):
    def __init__(self, data_path, sample_num, first_n_byte, padding_value=0):
        self.data_path = data_path
        self.fp_list = random.sample(os.listdir(data_path), sample_num)
        #self.labels = pd.read_csv('resource/total.csv').set_index(['name'])
        self.labels = pd.read_csv('resource/labels.csv').set_index(['hash'])
        self.first_n_byte = first_n_byte
        self.padding_value = padding_value
        if padding_value == 256:
            self.bias = 0
        else:
            self.bias = 1

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        #class_label = self.labels.loc[self.fp_list[idx]]['is_malware']
        class_label = self.labels.loc[self.fp_list[idx].split('.')[0]]['is_malware']

        with open(self.data_path+self.fp_list[idx],'rb') as f:
            origin = np.array([i+self.bias for i in f.read()[:self.first_n_byte]])
            origin_len = origin.size
            
            if self.padding_value == 0:
                origin = np.concatenate((origin, np.zeros(self.first_n_byte-origin_len)), axis=0)     
            else:
                origin  = np.concatenate((origin, np.ones(self.first_n_byte-origin_len, dtype=np.uint16)*self.padding_value), axis=0)
        return torch.Tensor(origin).long(), torch.Tensor([class_label]).long()

    
        

class InsertDataset(Dataset):
    def __init__(self, fp_list, data_path, insert_size, first_n_byte=1000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.insert_size = insert_size
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        with open(self.data_path+self.fp_list[idx],'rb') as f:
            origin = np.array([i+1 for i in f.read()[:self.first_n_byte]])
            origin_len = origin.size
            
            adv = np.subtract(origin, np.ones_like(origin))
            adv, insert_offset = insert_SLACK(self.data_path+self.fp_list[idx], adv, self.insert_size)
            adv = np.add(adv, np.ones_like(adv))
            
            origin = np.concatenate((origin, np.zeros(self.first_n_byte-origin_len)), axis=0)     
            adv = np.concatenate( (adv, np.zeros(self.first_n_byte-len(adv))), axis=0 )

        return torch.Tensor(adv).long(), torch.Tensor([insert_offset]).long(), torch.Tensor(origin).long(), torch.Tensor([origin_len]).long()


class AppendDataset(Dataset):
    def __init__(self, config):
        file_names = []
        for f in os.listdir(config.sample_dir):
            location = os.path.join(config.sample_dir, f)
            if os.stat(location).st_size < config.first_n_byte - config.padding_len:
                file_names.append(f)

        self.fp_list = file_names
        self.data_path = config.sample_dir
        self.padding_len = config.padding_len
        self.first_n_byte = config.first_n_byte
        self.padding_value = config.padding_value

        if config.padding_value == 0:
            self.bias = 1
        else:
            self.bias = 0

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        with open(self.data_path+self.fp_list[idx],'rb') as f:
            origin = np.array([i+self.bias for i in f.read()])
            origin_len = origin.size

            if self.padding_value == 0:
                tmp = np.concatenate((origin, np.zeros(self.first_n_byte-origin_len)), axis=0)
                adv = np.concatenate((origin, np.random.randint(1, 257, self.padding_len), np.zeros(self.first_n_byte-origin_len-self.padding_len)), axis=0 )
            else:
                tmp = np.concatenate((origin, np.ones(self.first_n_byte-origin_len, dtype=np.uint16)*self.padding_value), axis=0)
                adv = np.concatenate((origin, np.random.randint(0, 256, self.padding_len), np.ones(self.first_n_byte-origin_len-self.padding_len, dtype=np.uint16)*self.padding_value), axis=0)

        return torch.Tensor(adv).long(), torch.Tensor(tmp).long(), torch.Tensor([origin_len]).long(), self.data_path+self.fp_list[idx]
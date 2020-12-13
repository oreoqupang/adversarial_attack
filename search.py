import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import os
from attack.CustomGradAppend import CustomGradAppendAttack
from src.markin_MalConv import MalConv_markin
from src.util import *
from src.Target import Target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('./malconv.pt'))

test_num = 50
batch_size = 8
loop_num = 1
first_n_byte = 1000000
padding_len = 1000
sample_dir = 'D:\\sub74\\malwares\\'
file_names = []
for f in os.listdir(sample_dir):
  location = os.path.join(sample_dir, f)
  if os.stat(location).st_size < first_n_byte - padding_len:
    file_names.append(f)

test_loader = DataLoader(AppendDataset(file_names, sample_dir, 
    padding_len, first_n_byte),
  batch_size=batch_size, shuffle=True)

target = Target(malconv, nn.BCEWithLogitsLoss(), F.relu, 0.5, first_n_byte)
attack = CustomGradAppendAttack(target, padding_len, malconv.byte_embedding, loop_num)

def test_function(a, b):
    test_cnt = 0
    attack_cnt = 0
    success_cnt = 0
    
    for bytez, ori_bytez, bytez_len, names in test_loader:
        if test_cnt >= test_num:
            break
        test_cnt += 1

        ori_bytez, bytez, bytez_len = ori_bytez.to(device), bytez.to(device), bytez_len.to(device)
        
        embed = attack.embedding(ori_bytez).detach()
        init_outputs = target.predict(embed)
        init_result = target.get_result(init_outputs)

        #init_result = torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte)
        attack_cnt += torch.count_nonzero(init_result)
        attack_bytez = bytez[init_result]
        result = attack.do_attack(attack_bytez, bytez_len[init_result], (a, b)) 

        success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
    
    return success_cnt/attack_cnt 
    
    

param_num  = 2
low_limits = [1 for i in range(param_num)]
high_limits = [100 for i in range(param_num)]
sample_num = 50
top_num = 5
T = 20

for i in range(T):
    samples = [[np.random.uniform(low_limits[i], high_limits[i]) for i in range(param_num)] for _ in range(sample_num)]
    results = []
    for idx, params in enumerate(samples):
        res = test_function(params[0], params[1])
        results.append(res)
        print(f"Test_{i} : sample #{idx}, value : {res}")

    top_idxs = np.argsort(np.array(results))[-top_num:]
    top_params = np.take(samples, top_idxs, axis=0)
    print(f"top{top_num} : {np.array(results)[top_idxs]}, params:{top_params}")
    candi_high_limits = top_params.max(axis=0)
    candi_low_limits = top_params.min(axis=0)
    
    for i in range(param_num):
        if low_limits[i] < candi_low_limits[i]:
            low_limits[i] = candi_low_limits[i]

        if high_limits[i] > candi_high_limits[i]:
            high_limits[i] = candi_high_limits[i]

    print(low_limits, high_limits)

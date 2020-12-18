import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import os
from attack.CustomGradAppend import CustomGradAppendAttack
from src.MalConv import MalConv1, MalConv2, MalConv3
from src.util import *
from src.Target import Target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_config = read_config('test.json')
malconv_config = read_config('malconv.json')

#model1
#model = MalConv1(channels=256, window_size=512, embd_size=8).to(device)
#model.load_state_dict(torch.load('malconv.checkpoint')['model_state_dict'])

#model2
#model = MalConv2().to(device)
#model.load_state_dict(torch.load('pretrained_malconv.pth'))

#model3
model = MalConv3(malconv_config).to(device)
model.load_state_dict(torch.load('./malconv.pt'))

model.eval()

test_loader = DataLoader(AppendDataset(test_config),
  batch_size=test_config.batch_size, shuffle=True)

target = target = Target(model, 0.5, test_config.first_n_byte)
attack = CustomGradAppendAttack(target, test_config.padding_len, model.get_embedding(), test_config.loop_num)

def test_function(a, b):
    test_cnt = 0
    attack_cnt = 0
    success_cnt = 0
    
    for bytez, ori_bytez, bytez_len, names in test_loader:
        if test_cnt >= test_config.test_num:
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
    with open("result/searching_log", "at") as f:
        f.write(f"\n\ntop{top_num} : {np.array(results)[top_idxs]}, params:{top_params}")

    candi_high_limits = top_params.max(axis=0)
    candi_low_limits = top_params.min(axis=0)
    
    for i in range(param_num):
        if low_limits[i] < candi_low_limits[i]:
            low_limits[i] = candi_low_limits[i]

        if high_limits[i] > candi_high_limits[i]:
            high_limits[i] = candi_high_limits[i]

    print(low_limits, high_limits)
    with open("result/searching_log_4", "at") as f:
        f.write(f"\n\n{low_limits}, {high_limits}")

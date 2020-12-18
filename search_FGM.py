import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from attack.FGM import FGMAttack
from src.MalConv import MalConv1, MalConv2, MalConv3
from src.util import *
from src.Target import Target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_config = read_config('test.json')
malconv_config = read_config('malconv.json')

#model1
model = MalConv1(channels=256, window_size=512, embd_size=8).to(device)
model.load_state_dict(torch.load('malconv.checkpoint')['model_state_dict'])

#model2
#model = MalConv2().to(device)
#model.load_state_dict(torch.load('pretrained_malconv.pth'))

#model3
#model = MalConv3(malconv_config).to(device)
#model.load_state_dict(torch.load('./malconv.pt'))

model.eval()
test_loader = DataLoader(AppendDataset(test_config),
  batch_size=test_config.batch_size, shuffle=True)

target = Target(model, 0.5, test_config.first_n_byte)
attack = FGMAttack(target, model.get_embedding(), test_config.padding_value)

def test_function(a):
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
        result = attack.do_append_attack(attack_bytez, bytez_len[init_result], test_config.padding_len, a) 

        success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
    
    return success_cnt/attack_cnt 

top_num = 10

M = model.get_embedding()(torch.arange(0, 257).to(device))
r = int(torch.max(M) - torch.min(M))
step = r/100
params = np.arange(step, r, step)
print(params)

results = []
for i, p in enumerate(params):
    res = test_function(p)
    results.append(res)
    print(f"Test_{i} : sample{p}, value : {res}")

top_idxs = np.argsort(np.array(results))[-top_num:]
top_params = np.take(params, top_idxs, axis=0)
print(f"top{top_num} : {np.array(results)[top_idxs]}, params:{top_params}")
with open("result/searching_log_FGM", "at") as f:
    f.write(f"\n\ntop{top_num} : {np.array(results)[top_idxs]}, params:{top_params}")
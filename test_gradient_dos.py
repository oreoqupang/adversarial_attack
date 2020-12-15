import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from src.MalConv import MalConv1, MalConv2, MalConv3
from src.util import *
from src.Target import Target
from torch.utils.data import DataLoader
from attack.Gradient import GradientAttack

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


attack = GradientAttack(target, test_config.padding_len, model.get_embedding(), test_config.loop_num, test_config.padding_value)

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
  print(f"Initial test({test_cnt}) result : {init_result}")

  attack_cnt += torch.count_nonzero(init_result)
  attack_bytez = bytez[init_result]
  result = attack.do_header_attack(attack_bytez, bytez_len[init_result]) 

  success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
  
print("--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"result/{test_config.test_name}({test_config.padding_len})", "at") as output:
  output.write(f"--------Test Reslut({test_config.loop_num})----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n\n\n")

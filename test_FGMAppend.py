import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from torch.utils.data import DataLoader
from src.MalConv import MalConv1, MalConv2, MalConv3
from src.Target import Target
from src.util import *
from attack.FGM import FGMAttack


def embedding_Mapping(embed_x):
  x_len = embed_x.size()[0]

  shortest_bytes = torch.argmin(torch.cdist(embed_x, M, p = 2), dim=-1)+1
  return shortest_bytes

def FGM_append_attack(bytez, bytez_len, result, grad, eps):
  for idx in range(len(bytez)):
    if result[idx] == False: #classfied benign or no padding
      continue

    origin_len = int(bytez_len[idx])
    padding = bytez[idx, origin_len:origin_len+padding_len]
    padding_grad = grad[idx, origin_len:origin_len+padding_len]

    z = embedding(padding)
    z = z - eps * padding_grad.detach().sign()
    bytez[idx, origin_len:origin_len+padding_len] = embedding_Mapping(z)


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

target = Target(model, 0.5, test_config.first_n_byte)
attack = FGMAttack(target, model.get_embedding(), test_config.padding_value)

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
  
  result = attack.do_append_attack(attack_bytez, bytez_len[init_result], test_config.padding_len, 0.9) 

  success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
  
  
print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

"""
with open(f"test_FGM_result({padding_len})", "wt") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n")
  """
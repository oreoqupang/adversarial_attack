import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from torch.utils.data import DataLoader
from src.MalConv import MalConv
from src.markin_MalConv import MalConv_markin
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

#malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
#malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])
config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('./malconv.pt'))

test_num = 30
batch_size = 8
first_n_byte = 1000000
padding_len = 1000

sample_dir = 'D:\\sub74\\malwares\\'
file_names = []
for f in os.listdir('D:\\sub74\\malwares\\'):
  location = os.path.join(sample_dir, f)
  if os.stat(location).st_size < first_n_byte - padding_len:
    file_names.append(f)

test_loader = DataLoader(AppendDataset(file_names, sample_dir, 
  padding_len, first_n_byte),
  batch_size=batch_size, shuffle=True)

target = Target(malconv, nn.BCEWithLogitsLoss(), F.relu, 0.5, first_n_byte)
attack = FGMAttack(target, malconv.byte_embedding)

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
  print(f"Initial test({test_cnt}) result : {init_result}")


  attack_cnt += torch.count_nonzero(init_result)
  attack_bytez = bytez[init_result]
  
  result = attack.do_append_attack(attack_bytez, bytez_len[init_result], padding_len, 0.9) 

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
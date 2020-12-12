import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from src.MalConv import MalConv
from src.markin_MalConv import MalConv_markin
from src.util import *
from src.Target import Target
from torch.utils.data import DataLoader
from attack.FGM import FGMAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
#malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])
config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('/home/choiwonseok/malconv.pt'))

test_num = 5
batch_size = 64
first_n_byte = 1000000

sample_dir = '/home/choiwonseok/sample/malwares/'
file_names = []
for f in os.listdir('/home/choiwonseok/sample/malwares'):
  location = os.path.join(sample_dir, f)
  if os.stat(location).st_size < first_n_byte:
    file_names.append(f)

sample_dir = '/home/choiwonseok/sample/malwares/'
test_loader = DataLoader(AppendDataset(file_names, "/home/choiwonseok/sample/malwares/", 
  0, first_n_byte),
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

  target_idxs = []
  for i in range(batch_size):
    if init_result[i]:
      idxs = find_SLACK(names[i])
      if len(idxs) == 0:
        init_result[i] = False
      else:
        target_idxs.append(find_SLACK(names[i]))

  attack_cnt += torch.count_nonzero(init_result)
  attack_bytez = bytez[init_result]
  print(len(attack_bytez), len(target_idxs))
    
  result = attack.do_slack_attack(attack_bytez, target_idxs, 0.1) 
  success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
  
print("--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

"""
with open(f"test_result({padding_limit})", "at") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n\n\n")
"""
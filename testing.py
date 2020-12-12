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
from attack.GradientAppend import GradientAppendAttack
from attack.FGMAppend import FGMAppendAttack
from attack.CustomGradAppend import CustomGradAppendAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
#malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])

config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('/home/choiwonseok/malconv.pt'))

test_num = 5
batch_size = 64
loop_num = 1
first_n_byte = 1000000
padding_limit = 1000

#file_names = os.listdir('/home/choiwonseok/sample/malwares')

sample_dir = '/home/choiwonseok/sample/malwares/'
file_names = []
for f in os.listdir('/home/choiwonseok/sample/malwares'):
  location = os.path.join(sample_dir, f)
  if os.stat(location).st_size < first_n_byte:# - padding_limit:
    file_names.append(f)
test_loader = DataLoader(Dataset(file_names, "/home/choiwonseok/sample/malwares/", 
  [1]*len(file_names), padding_limit, first_n_byte),
  batch_size=batch_size, shuffle=True)
"""
test_loader = DataLoader(ExeDataset_FGM(file_names, sample_dir, 
  [1]*len(file_names), padding_limit, first_n_byte),
  batch_size=batch_size, shuffle=True)
"""

target = Target(malconv, nn.BCEWithLogitsLoss(), F.relu, 0.5, first_n_byte)
#attack = GradientAppendAttack(target, padding_limit, malconv.byte_embedding, loop_num)
#attack = FGMAppendAttack(target, padding_limit, malconv.byte_embedding)
attack = CustomGradAppendAttack(target, padding_limit, malconv.byte_embedding, loop_num)

test_cnt = 0
attack_cnt = 0
success_cnt = 0

for bytez, ori_bytez, bytez_len in test_loader:
  if test_cnt >= test_num:
    break
  test_cnt += 1

  ori_bytez, bytez, bytez_len = ori_bytez.to(device), bytez.to(device), bytez_len.to(device)
  
  embed = attack.embedding(ori_bytez).detach()
  init_outputs = target.predict(embed)
  init_result = target.get_result(init_outputs)
  print(f"Initial test({test_cnt}) result : {init_result}")

  #init_result = torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte)
  attack_cnt += torch.count_nonzero(init_result)
  attack_bytez = bytez[init_result]
  print(attack_bytez.size(), attack_cnt)
  result = attack.do_attack(attack_bytez, bytez_len[init_result], (78.2, 8.8)) 

  success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))
  
print("--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_result({padding_limit}_custom)", "at") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n\n\n")
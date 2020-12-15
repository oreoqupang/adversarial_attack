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
from attack.CustomGradAppend import CustomGradAppendAttack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
#malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])
config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('./malconv.pt'))

test_num = 10
batch_size = 64
loop_num = 1
first_n_byte = 1000000
padding_len = 1000
params =  (66.6, 2.2) #(78.2, 8.8) #[68.9,  7.8]
sample_dir =  '/home/choiwonseok/sample/malwares/'
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
  result = attack.do_attack(attack_bytez, bytez_len[init_result],  params)
  #(68.9, 7.8)) 

  success_cnt += (torch.count_nonzero(init_result) - torch.count_nonzero(result))#(78.2, 8.8))
  
print("--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"result/test_result({padding_len}_custom)", "at") as output:
  output.write(f"--------Test Reslut({params})----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n\n\n")

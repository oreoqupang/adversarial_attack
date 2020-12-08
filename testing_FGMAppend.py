import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from src.MalConv import MalConv
from src.util import ExeDataset_FGM
from torch.utils.data import DataLoader


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

malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])

test_num = 5
batch_size = 128
loop_num = 1
first_n_byte = 1000000
padding_len = 10000
sample_dir = '/home/choiwonseok/sample/malwares/'
file_names = []
for f in os.listdir('/home/choiwonseok/sample/malwares'):
  location = os.path.join(sample_dir, f)
  if os.stat(location).st_size < first_n_byte - padding_len:
    file_names.append(f)
print(len(file_names))
test_loader = DataLoader(ExeDataset_FGM(file_names, sample_dir, 
  [1]*len(file_names), padding_len, first_n_byte),
  batch_size=batch_size, shuffle=True)

embedding = malconv.embd
M = embedding(torch.arange(1, 257).to(device))

test_cnt = 0
attack_cnt = 0
success_cnt = 0
lables = torch.tensor([0]*batch_size, dtype=torch.long).to(device)

for bytez, ori_bytez, label, bytez_len in test_loader:
  if test_cnt >= test_num:
    break
  test_cnt += 1

  bytez, ori_bytez, label, bytez_len = bytez.to(device), ori_bytez.to(device), label.to(device), bytez_len.to(device)
  
  embed = embedding(ori_bytez).detach()
  init_outputs = malconv(embed)
  init_malness = F.softmax(init_outputs, dim=-1)
  init_result = init_malness[:,1] > 0.5
 
  print(f"Initial test({test_cnt}) result : {init_result}")
  for i in range(loop_num):
    print(f"try loop {i}")
    if i == 0:
      attack_cnt += torch.count_nonzero(init_result)

    
    embed = embedding(bytez).detach()
    embed.requires_grad = True
    malconv.zero_grad()

    outputs = malconv(embed)
    loss = nn.CrossEntropyLoss()(outputs, lables)
    malness = F.softmax(outputs, dim=-1)
    result = torch.logical_and(malness[:,1] > 0.5, init_result)
   
    loss.backward()
    FGM_append_attack(bytez, bytez_len, result, embed.grad, 0.7)
    
  success_cnt += torch.count_nonzero(init_result) - torch.count_nonzero(result)
  
print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_FGM_result({padding_len})", "wt") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n")
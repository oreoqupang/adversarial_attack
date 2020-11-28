import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import struct
import os
from src.MalConv import MalConv
from src.util import ExeDataset
from torch.utils.data import DataLoader

def prepareData(data_loader, model, embedding):
    out = torch.empty()

def do_attack(bytez, bytez_len, result, gradient):
  for idx in range(len(bytez)):
    if result[idx] == 0 or bytez_len[idx] >= first_n_byte: #classfied benign or no padding
      continue
    
    byte_len = int(bytez_len[idx])
    padding_len = min(byte_len + padding_limit, first_n_byte) - byte_len
    padding = bytez[idx, byte_len:byte_len+padding_len]

    for j in range(padding_len):
      x_j = padding[j]
      z_j = M[x_j]

      w_j = -gradient[idx, j+byte_len]
      if torch.norm(w_j, p=2) != 0:
        n_j =  w_j / torch.norm(w_j, p=2)
      else:
        continue

      max_distance = -1
      max_value = None
      for i in range(256):
        s_i = n_j.dot(M[i]-z_j)
        if s_i > 0:
          d_i = torch.norm(M[i]-(z_j + s_i*n_j), p=2)
          if max_distance < d_i:
            max_distance = d_i
            max_value = i

      if max_value != None:
        padding[j] = max_value

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
weights = torch.load('./malconv.checkpoint', map_location='cpu')
malconv.load_state_dict(weights['model_state_dict'])
malconv.eval()

test_num = 2
loop_num = 20
first_n_byte = 1000000
padding_limit = 1000
file_names = os.listdir('/Users/choiwonseok/samples/malwares')

test_loader = DataLoader(ExeDataset(file_names, "/Users/choiwonseok/samples/malwares/", 
  [1]*11000, padding_limit, first_n_byte),
  batch_size=4, shuffle=True)

embedding = malconv.embd
M = embedding(torch.arange(0, 257))

test_cnt = 0
attack_cnt = 0
success_cnt = 0

for bytez, label, bytez_len in test_loader:
  if test_cnt >= test_num:
    break
  test_cnt += 1

  bytez, label, bytez_len = bytez.to(device), label.to(device), bytez_len.to(device)
  
  embed = embedding(bytez).detach()
  embed.requires_grad = True
  init_outputs = malconv(embed)
  init_malness = F.softmax(init_outputs, dim=-1)[:, 1]
  init_result = init_malness > 0.5 
  test_idx = None

  print(f"Initial test({test_cnt}) result : {init_malness}")

  for i in range(loop_num):
    print(f"try loop {i}")
    if i == 0:
      attack_cnt += torch.count_nonzero(torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte))
      result = init_result
      init_malness.backward(torch.Tensor([1]*4))

    do_attack(bytez, bytez_len, result, embed.grad)

    embed = embedding(bytez).detach()
    embed.requires_grad = True
    outputs = malconv(embed)
    malness = F.softmax(outputs, dim=-1)[:, 1]
    result = malness > 0.5
    print(malness)

    malness.backward(torch.Tensor([1]*4))

  success_cnt += torch.count_nonzero(init_result) - torch.count_nonzero(result)

print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_result({padding_limit})", "wt") as output:
  output.write("--------Test Reslut----------")
  output.write(f"Total Attack : {attack_cnt}")
  output.write(f"Success count : {success_cnt}")
  output.write(f"SR : {success_cnt/attack_cnt*100}%")
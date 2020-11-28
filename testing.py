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
    with torch.no_grad():
      byte_len = int(bytez_len[idx])
      padding_len = min(byte_len + padding_limit, first_n_byte) - byte_len
      padding = bytez[idx, byte_len:byte_len+padding_len]

      z = embedding(padding) # padding_len * 8
      w = -gradient[idx, byte_len:byte_len+padding_len] # padding_len * 8
      n = w / torch.norm(w, dim=1, p=2)[:, None] # padding_len * 8

      ss=torch.bmm(torch.stack([M]*padding_len, dim = 0)-z.unsqueeze(dim=1), n.unsqueeze(dim=2)).squeeze() # ss size = padding_len * 256
      
      ds = torch.stack([M]*padding_len, dim = 0)-(torch.bmm(ss.view(padding_len, -1, 1), n.view(padding_len, 1, -1))+z.unsqueeze(dim=1))
      ds = torch.norm(ds, dim = 2, p = 2)
      
      filtered_ds = ((ss>0) * ds)
      bytez[idx, byte_len:byte_len+padding_len] = torch.where(torch.isnan(filtered_ds[:, 0]), padding, filtered_ds.argmax(dim=1)+1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
weights = torch.load('./malconv.checkpoint', map_location='cpu')
malconv.load_state_dict(weights['model_state_dict'])
malconv.eval()

test_num = 4
batch_size = 64
loop_num = 20
first_n_byte = 1000000
padding_limit = 1000
file_names = os.listdir('/home/choiwonseok/sample/malwares_2017')

test_loader = DataLoader(ExeDataset(file_names, "/home/choiwonseok/sample/malwares_2017/", 
  [1]*11000, padding_limit, first_n_byte),
  batch_size=batch_size, shuffle=True)

embedding = malconv.embd
M = embedding(torch.arange(1, 257))

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
      init_malness.backward(torch.Tensor([1]*batch_size))

    do_attack(bytez, bytez_len, result, embed.grad)

    embed = embedding(bytez).detach()
    embed.requires_grad = True
    outputs = malconv(embed)
    malness = F.softmax(outputs, dim=-1)[:, 1]
    result = malness > 0.5
    print(malness)

    malness.backward(torch.Tensor([1]*batch_size))

  success_cnt += torch.count_nonzero(init_result) - torch.count_nonzero(result)

print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_result({padding_limit})", "wt") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n")
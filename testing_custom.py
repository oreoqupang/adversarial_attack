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


def do_attack(bytez, bytez_len, result, gradient, a1, a2):
  for idx in range(len(bytez)):
    if result[idx] == False or int(bytez_len[idx]) >= first_n_byte: #classfied benign or no padding
      continue

    with torch.no_grad():
      
      byte_len = int(bytez_len[idx])
      padding_len = min(byte_len + padding_limit, first_n_byte) - byte_len
      max_values = torch.tensor([[np.infty]*256]*padding_len, dtype=torch.float).to(device)
      """
      padding = bytez[idx, byte_len:byte_len+padding_len]

      z = embedding(padding) # padding_len * 8
      w = -gradient[idx, byte_len:byte_len+padding_len] # padding_len * 8
      n = w / torch.norm(w, dim=1, p=2)[:, None] # padding_len * 8

      ss=torch.bmm(torch.stack([M]*padding_len, dim = 0)-z.unsqueeze(dim=1), n.unsqueeze(dim=2)).squeeze() # ss size = padding_len * 256
      ds_1 = torch.stack([M]*padding_len, dim = 0)-(torch.bmm(ss.view(padding_len, -1, 1), n.view(padding_len, 1, -1))+z.unsqueeze(dim=1))
      ds_2 = torch.cdist(z.unsqueeze(dim=1), torch.stack([M]*padding_len, dim=0)).squeeze(dim=1) #padding_len * 1 * 256
      ds = torch.norm(ds_1, dim = 2, p = 2) * a1 + ds_2 * a2

      if not (torch.equal( (torch.norm(w,dim=1,p=2) == 0), torch.isnan(ds[:,0]))):
        print("something wrong")
        sys.exit(-1)

      filtered_ds = torch.where(ss>0, ds, max_values)
      """
      random_padding = torch.Tensor(np.random.randint(1, 257, padding_len))
      bytez[idx, byte_len:byte_len+padding_len] = random_padding
      #bytez[idx, byte_len:byte_len+padding_len] = torch.where(torch.isnan(ds[:, 0]), padding, filtered_ds.argmin(dim=1)+1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])

test_num = 5
batch_size = 128
loop_num = 2
first_n_byte = 1000000
padding_limit = 500
file_names = os.listdir('/home/choiwonseok/sample/malwares')

test_loader = DataLoader(ExeDataset(file_names, "/home/choiwonseok/sample/malwares/", 
  [1]*len(file_names), padding_limit, first_n_byte),
  batch_size=batch_size, shuffle=True)

embedding = malconv.embd
M = embedding(torch.arange(1, 257).to(device))

test_cnt = 0
attack_cnt = 0
success_cnt = 0
random_succ_cnt = 0
lables = torch.tensor([0]*batch_size, dtype=torch.long).to(device)

for bytez, label, bytez_len in test_loader:
  if test_cnt >= test_num:
    break
  test_cnt += 1

  bytez, label, bytez_len = bytez.to(device), label.to(device), bytez_len.to(device)
  
  embed = embedding(bytez).detach()
  embed.requires_grad = True
  init_outputs = malconv(embed)
  init_malness = F.softmax(init_outputs, dim=-1)
  
  loss = nn.CrossEntropyLoss()(init_outputs, lables)
  init_result = init_malness[:,1] > 0.5
  test_idx = None
  
  print(f"Initial test({test_cnt}) result : {init_result}")
  for i in range(loop_num):
    print(f"try loop {i}")
    if i == 0:
      attack_cnt += torch.count_nonzero(torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte))
      result = init_result
      loss.backward()

    do_attack(bytez, bytez_len, result, embed.grad, 0.5, 0.5)

    embed = embedding(bytez).detach()
    embed.requires_grad = True
    malconv.zero_grad()

    outputs = malconv(embed)
    loss = nn.CrossEntropyLoss()(outputs, lables)
    malness = F.softmax(outputs, dim=-1)
    result = malness[:,1] > 0.5
   
    loss.backward()

  success_cnt += torch.count_nonzero(torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte)) - torch.count_nonzero(torch.logical_and(result, bytez_len.squeeze() < first_n_byte))
  
print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_custom_result({padding_limit})", "wt") as output:
  output.write("--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n")
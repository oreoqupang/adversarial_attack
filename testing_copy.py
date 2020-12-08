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
from torch.utils.data import DataLoader


def do_attack(bytez, bytez_len, result, gradient):
  for idx in range(len(bytez)):
    if result[idx] == False or int(bytez_len[idx]) >= first_n_byte: #classfied benign or no padding
      continue

    with torch.no_grad():
      byte_len = int(bytez_len[idx])
      padding_len = min(byte_len + padding_limit, first_n_byte) - byte_len
      max_values = torch.tensor([[np.infty]*256]*padding_len, dtype=torch.float).to(device)
      padding = bytez[idx, byte_len:byte_len+padding_len]

      z = embedding(padding) # padding_len * 8
      w = -gradient[idx, byte_len:byte_len+padding_len] # padding_len * 8
      n = w / torch.norm(w, dim=1, p=2)[:, None] # padding_len * 8

      ss=torch.bmm(torch.stack([M]*padding_len, dim = 0)-z.unsqueeze(dim=1), n.unsqueeze(dim=2)).squeeze() # ss size = padding_len * 256
      ds = torch.stack([M]*padding_len, dim = 0)-(torch.bmm(ss.view(padding_len, -1, 1), n.view(padding_len, 1, -1))+z.unsqueeze(dim=1))
      ds = torch.norm(ds, dim = 2, p = 2)
      if not (torch.equal( (torch.norm(w,dim=1,p=2) == 0) ,  torch.isnan(ds[:,0]))):
        print("something wrong")
        sys.exit(-1)

      filtered_ds = torch.where(ss>0, ds, max_values)
      ttmp = torch.where(torch.isnan(ds[:, 0]), padding, filtered_ds.argmin(dim=1)+1)
      bytez[idx, byte_len:byte_len+padding_len] = ttmp
      
      #random_padding = torch.Tensor(np.random.randint(1, 257, padding_len))
      #bytez[idx, byte_len:byte_len+padding_len] = random_padding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
#malconv.load_state_dict(torch.load(('./malconv.checkpoint'))['model_state_dict'])

config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('/home/choiwonseok/malconv.pt'))

test_num = 1
batch_size = 4
loop_num = 2
first_n_byte = 1000000
padding_limit = 10000
file_names = os.listdir('/home/choiwonseok/sample/malwares')

test_loader = DataLoader(ExeDataset(file_names, "/home/choiwonseok/sample/malwares/", 
  [1]*len(file_names), padding_limit, first_n_byte),
  batch_size=batch_size, shuffle=True)

#embedding = malconv.embd
embedding = malconv.byte_embedding
M = embedding(torch.arange(1, 257).to(device))

test_cnt = 0
attack_cnt = 0
success_cnt = 0
random_succ_cnt = 0
lables = torch.tensor([[0.0]]*batch_size, dtype=torch.float).to(device)

for bytez, ori_bytez, label, bytez_len in test_loader:
  if test_cnt >= test_num:
    break
  test_cnt += 1

  ori_bytez, bytez, label, bytez_len = ori_bytez.to(device), bytez.to(device), label.to(device), bytez_len.to(device)
  
  embed = embedding(ori_bytez).detach()
  init_outputs = malconv(embed)
  #init_malness = F.softmax(init_outputs, dim=-1)
  #init_result = init_malness[:,1] > 0.5
  init_result = (nn.Sigmoid()(init_outputs) > 0.5).squeeze()


  print(f"Initial test({test_cnt}) result : {init_result}")
  for i in range(loop_num):
    print(f"try loop {i}")
    if i == 0:
      attack_cnt += torch.count_nonzero(torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte))

    embed = embedding(bytez).detach()
    embed.requires_grad = True

    outputs = malconv(embed)
    #loss = nn.CrossEntropyLoss()(outputs, lables)
    #malness = F.softmax(outputs, dim=-1)
    #result = torch.logical_and(malness[:,1] > 0.5, init_result)
    loss = nn.BCEWithLogitsLoss()(outputs, lables)
    result = (nn.Sigmoid()(outputs) > 0.5).squeeze()
    loss.backward()
    #back = bytez.clone()
    do_attack(bytez, bytez_len, torch.logical_and(result, init_result), embed.grad)
    """
    for j in range(len(bytez)):
      if not(torch.equal(back[j, bytez_len[j]+padding_limit:], bytez[j, bytez_len[j]+padding_limit:]) and torch.equal(back[j, :bytez_len[j]], bytez[j, :bytez_len[j]])):
        print("fuck")
        sys.exit(-1)

    embed2 = embedding(back).detach()
    embed2.requires_grad = True

    outputs = malconv(embed2)
    
    loss = nn.CrossEntropyLoss()(outputs, lables)
    malness2 = F.softmax(outputs, dim=-1)
    print(malness, malness2)
    """

  success_cnt += torch.count_nonzero(torch.logical_and(init_result, bytez_len.squeeze() < first_n_byte)) - torch.count_nonzero(torch.logical_and(result, bytez_len.squeeze() < first_n_byte))
  for i in range(batch_size):
    if init_result[i] and bytez_len[i] < first_n_byte and result[i] == False:
      padding_len = int(min(bytez_len[i] + padding_limit, first_n_byte) - bytez_len[i])
      (bytez[i, :bytez_len[i]+padding_len]-1).cpu().detach().numpy().astype(np.int8).tofile(f"adv_{padding_len}_{i}", sep="")
      (ori_bytez[i, :bytez_len[i]]-1).cpu().detach().numpy().astype(np.int8).tofile(f"origin_{padding_len}_{i}")
      print(nn.Sigmoid()(init_outputs), nn.Sigmoid()(outputs))




print("\n\n--------Test Reslut----------")
print(f"Total Attack : {attack_cnt}")
print(f"Success count : {success_cnt}")
print(f"SR : {success_cnt/attack_cnt*100}%")

with open(f"test_result({padding_limit})", "at") as output:
  output.write("\n\n--------Test Reslut----------\n")
  output.write(f"Total Attack : {attack_cnt}\n")
  output.write(f"Success count : {success_cnt}\n")
  output.write(f"SR : {success_cnt/attack_cnt*100}%\n")
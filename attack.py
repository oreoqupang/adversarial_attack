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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
weights = torch.load('./malconv.checkpoint', map_location='cpu')
malconv.load_state_dict(weights['model_state_dict'])
malconv.eval()

file_names = os.listdir('/home/choiwonseok/sample/malwares_2017')
test_loader = DataLoader(ExeDataset(file_names, "/home/choiwonseok/sample/malwares_2017/", [1]*11000),
  batch_size=1, shuffle=True)

test_num = 400
loop_num = 20
blen_limit = 1000000
padding_limit = 1000

embedding = malconv.embd
M = embedding(torch.arange(0, 256))

test_cnt = 0
attack_cnt = 0
success_cnt = 0
for bytez, label in test_loader:
  if test_cnt >= test_num:
     break

  binary_len = len(bytez[0])
  padding_bytes = min(binary_len + padding_limit, blen_limit) - binary_len
  if padding_bytes == 0:
    continue

  print(f"Test {test_cnt} start")
  
  bytez, label = bytez.to(device), label.to(device)
  padding = np.random.randint(0, 256, padding_bytes)

  init_outputs = malconv(embedding(bytez))
  init_result = F.softmax(init_outputs, dim=-1)
    
  malicious_ness = init_result[0, 1]
  if malicious_ness < 0.5:
    continue

  test_cnt+=1
  print(f"attack start {binary_len}, {padding_bytes}")
  for t in range(loop_num):
    inp = torch.cat([bytez, torch.from_numpy(padding[np.newaxis, :]).to(device)], dim=1).float()
    inp_adv = inp.requires_grad_()
    emb_adv = embedding(inp_adv.long()).detach()
    emb_adv.requires_grad = True

    outputs = malconv(emb_adv)
    result = F.softmax(outputs, dim=-1)
    
    malicious_ness = result[0, 1]
    print(emb_adv.size(), malicious_ness)

    if malicious_ness < 0.5:
      print("success!!")
      success_cnt += 1

      break

    malicious_ness.backward()
    for j in range(padding_bytes):
      x_j = padding[j]
      z_j = M[x_j]

      w_j = -emb_adv.grad[0, j+binary_len] #compute gradient w_j 
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


print(f"success_count : {success_cnt}, SR : {success_cnt/400}")
import torch
import torch.nn.functional as F
import torch.nn as nn
from MalConv import MalConv
import numpy as np
import sys
import struct
import os


malconv = MalConv(channels=256, window_size=512, embd_size=8)
weights = torch.load('./malconv.checkpoint', map_location='cpu')
malconv.load_state_dict(weights['model_state_dict'])

with open('./gdrive/MyDrive/ML/aaa', 'rb') as infile:
  bytez = infile.read()

loop_num = 20
binary_len = len(bytez)
blen_limit = 1000000
padding_limit = 1000
padding_bytes = min(binary_len + padding_limit, blen_limit) - binary_len

embedding = malconv.embd
M = embedding(torch.arange(0, 256))

x = np.frombuffer(bytez, dtype=np.uint8)
padding = np.random.randint(0, 256, padding_bytes)
print(binary_len, padding_bytes)

for t in range(loop_num):
  inp = torch.from_numpy(np.concatenate([x, padding])[np.newaxis, :]).float()
  inp_adv = inp.requires_grad_() ## why?
  emb_adv = embedding(inp_adv.long()).detach()
  emb_adv.requires_grad = True
  outputs = malconv(emb_adv)
  result = F.softmax(outputs, dim=-1)
  
  malicious_ness = result[0, 1]
  print(emb_adv.size(), malicious_ness)
  if malicious_ness < 0.5:
    print("success!!")
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

if t == 20:
  print("fail")
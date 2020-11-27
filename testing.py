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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

malconv = MalConv(channels=256, window_size=512, embd_size=8).to(device)
weights = torch.load('./malconv.checkpoint', map_location='cpu')
malconv.load_state_dict(weights['model_state_dict'])
malconv.eval()

file_names = os.listdir('/home/choiwonseok/sample/malwares_2017')
test_loader = DataLoader(ExeDataset(file_names, "/home/choiwonseok/sample/malwares_2017/", [1]*11000),
  batch_size=128, shuffle=True)

test_num = 4
loop_num = 20
blen_limit = 1000000
padding_limit = 1000

embedding = malconv.embd
M = embedding(torch.arange(0, 256))

test_cnt = 0
attack_cnt = 0
success_cnt = 0
for bytez, label, byte_len in test_loader:
  bytez, label, byte_len = bytez.to(device), label.to(device), byte_len.to(device)
  print(bytez, bytez.size())

  init_outputs = malconv(embedding(bytez.long()))
  init_result = F.softmax(init_outputs, dim=-1)
  print(init_result[:, 1], torch.count_nonzero(init_result[:, 1] > 0.5))
 
  break
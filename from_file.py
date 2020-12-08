import torch
import torch.nn as nn
import sys
from src.markin_MalConv import MalConv_markin
from src.util import *

file_name = "./adv_10000_2"
file_name2 = "./origin_10000_2"
with open(file_name, "rb") as f:
    raw_bytes = np.array([i+1 for i in f.read()], dtype=np.uint8)

with open(file_name2, "rb") as f:
    raw_bytes2 = np.array([i+1 for i in f.read()], dtype=np.uint8)

for i in range(len(raw_bytes)):
    if raw_bytes[i] != raw_bytes2[i]:
        print(i)
        break
sys.exit(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_config('malconv.json')
malconv = MalConv_markin(config).to(device)
malconv.load_state_dict(torch.load('/home/choiwonseok/malconv.pt'))

embedding = malconv.byte_embedding

inp = torch.from_numpy(np.concatenate([raw_bytes, np.zeros(1000000-len(raw_bytes))])[np.newaxis, :]).to(device)
print(len(inp[0]))
#inp_adv = inp.requires_grad_().to(device)
embd_x = embedding(inp.long()).detach()
embd_x.requires_grad = True
outputs = malconv(embd_x)
results = nn.Sigmoid()(outputs)

print(results)
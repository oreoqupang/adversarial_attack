import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.util import *
from src.Malconv import MalConv
from src.Malconv2 import MalConv2
from src.Malconv3 import MalConv3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    config = read_config('malconv.json')

    # Define Data
    pe_dataset = PEDataset(config.data_dir, 5000, config.first_n_byte, padding_value = 0)
    pe_loader = DataLoader(pe_dataset, batch_size=32, shuffle=False)

    # Define Model & Optimization Scheme

    #model = MalConv(channels=256, window_size=512, embd_size=8).to(device)
    #model.load_state_dict(torch.load('malconv.checkpoint')['model_state_dict'])

    #model = MalConv2().to(device)
    #model.load_state_dict(torch.load('pretrained_malconv.pth'))

    model = MalConv3(config).to(device)
    model.load_state_dict(torch.load('./malconv.pt'))
    model.eval()

    TP, FP, FN = 0, 0, 0
    precisions = []
    recalls = []
    for test_num, (test_X, test_y) in enumerate(pe_loader):
        test_X, test_y = test_X.to(device), test_y.to(device)
        probs = model(test_X)
        test_y = test_y.squeeze()
        #case Malconv
        #result = F.softmax(probs, dim=1)[:, 1] > 0.5
        #print(result, test_y)

        #case Malconv2
        #result = probs.squeeze() > 0.5
        #print(result, test_y)

        # case MalConv3
        result = torch.sigmoid(probs).squeeze() > 0.5
        print(result)
        TP += (torch.logical_and(result == True, test_y == 1)).sum().item()
        FP += (torch.logical_and(result == True, test_y == 0)).sum().item()
        FN += (torch.logical_and(result == False, test_y == 1)).sum().item()
        print(f'#{test_num} : {TP}, {FP}, {FN}')

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    print(f'test{config.model_name} - precision:{precision}, recall:{recall}')

with open("eval_result", "at") as f:
    f.write(f'\n{config.model_name}\n')
    f.write(f'precision:{precision}, recall:{recall}\n')
    f.write(f'TP:{TP}, FP:{FP}, FN:{FN}\n')
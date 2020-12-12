import torch
import numpy as np
import sys


device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomGradAppendAttack:
    def __init__(self, target, padding_limit, embedding, iter_num):
        self.target = target
        self.embedding = embedding
        self.padding_limit = padding_limit
        self.iter_num = iter_num

        self.M = embedding(torch.arange(1, 257).to(device))

    def update_bytes(self, bytez, bytez_len, gradient, result, params):
        for idx in range(len(bytez)):
            if result[idx] == False or int(bytez_len[idx]) >= self.target.first_n_byte:
                continue

            with torch.no_grad():
                byte_len = int(bytez_len[idx])
                padding_len = min(byte_len + self.padding_limit, self.target.first_n_byte) - byte_len
                max_values = torch.tensor([[np.infty]*256]*padding_len, dtype=torch.float).to(device)
                
                padding = bytez[idx, byte_len:byte_len+padding_len]
                z = self.embedding(padding) # padding_len * 8
                w = -gradient[idx, byte_len:byte_len+padding_len] # padding_len * 8
                n = w / torch.norm(w, dim=1, p=2)[:, None] # padding_len * 8

                ss = torch.bmm(torch.stack([self.M]*padding_len, dim = 0)-z.unsqueeze(dim=1), n.unsqueeze(dim=2)).squeeze() # ss size = padding_len * 256
                ds_1 = torch.stack([self.M]*padding_len, dim = 0)-(torch.bmm(ss.view(padding_len, -1, 1), n.view(padding_len, 1, -1))+z.unsqueeze(dim=1))
                ds_2 = torch.cdist(z.unsqueeze(dim=1), torch.stack([self.M]*padding_len, dim=0)).squeeze(dim=1) #padding_len * 1 * 256
                ds = torch.norm(ds_1, dim = 2, p = 2) * params[0] + ds_2 * params[1]
                if not (torch.equal( (torch.norm(w,dim=1,p=2) == 0) ,  torch.isnan(ds[:,0]))):
                    print("something wrong")
                    continue

                filtered_ds = torch.where(ss>0, ds, max_values)
                ttmp = torch.where(torch.isnan(ds[:, 0]), padding, filtered_ds.argmin(dim=1)+1)
                bytez[idx, byte_len:byte_len+padding_len] = ttmp
            

    def do_attack(self, bytez, bytez_len, params):
        for i in range(self.iter_num):
            embed = self.embedding(bytez).detach()
            embed.requires_grad = True

            outputs = self.target.predict(embed)
        
            labels = torch.zeros_like(outputs).to(device)
            loss = self.target.loss_function(outputs, labels)
            loss.backward()

            result = self.target.get_result(outputs)
            grads = embed.grad
            self.update_bytes(bytez, bytez_len, grads, result, params)
        
        embed = self.embedding(bytez).detach()
        final_outputs = self.target.predict(embed)
        return self.target.get_result(final_outputs)

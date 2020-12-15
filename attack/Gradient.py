import torch
import numpy as np
import sys

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradientAttack():
    def __init__(self, target, padding_limit, embedding, iter_num, padding_value=0):
        self.target = target
        self.embedding = embedding
        self.padding_limit = padding_limit
        self.iter_num = iter_num
        self.padding_value = padding_value
        
        if padding_value == 0:
            self.M = embedding(torch.arange(1, 257).to(device))
            self.bias = 1
        else:
            self.M = embedding(torch.arange(0, 256).to(device))
            self.bias = 0

    def update_bytes(self, bytez, bytez_len, gradient, result, target_idxs):
        for idx in range(len(bytez)):
            if result[idx] == False or int(bytez_len[idx]) >= self.target.first_n_byte:
                continue

            with torch.no_grad():
                padding_len = len(target_idxs[idx])
                max_values = torch.tensor([[np.infty]*256]*padding_len, dtype=torch.float).to(device)
                
                padding = bytez[idx, target_idxs[idx]]
                z = self.embedding(padding) # padding_len * 8
                w = -gradient[idx][target_idxs[idx]] # padding_len * 8
                n = w / torch.norm(w, dim=1)[:, None] # padding_len * 8

                ss = torch.bmm(torch.stack([self.M]*padding_len, dim = 0)-z.unsqueeze(dim=1), n.unsqueeze(dim=2)).squeeze() # ss size = padding_len * 256
                ds = torch.stack([self.M]*padding_len, dim = 0)-(torch.bmm(ss.view(padding_len, -1, 1), n.view(padding_len, 1, -1))+z.unsqueeze(dim=1))
                ds = torch.norm(ds, dim = 2, p = 'fro')
        
                if not (torch.equal( (torch.norm(w,dim=1) == 0) ,  torch.isnan(ds[:,0]))):
                    print("something wrong")
                    continue

                filtered_ds = torch.where(ss>0, ds, max_values)
                ttmp = torch.where(torch.isnan(ds[:, 0]), padding, filtered_ds.argmin(dim=1)+self.bias)
                bytez[idx, target_idxs[idx]] = ttmp

    def do_append_attack(self, bytez, bytez_len):
        target_idxs = []
        for idx in range(len(bytez)):
            byte_len = int(bytez_len[idx])
            padding_len = min(byte_len + self.padding_limit, self.target.first_n_byte) - byte_len
            target_idxs.append(torch.tensor(list(range(byte_len, byte_len+padding_len)), dtype=torch.long))    
            
        for i in range(self.iter_num):
            embed = self.embedding(bytez).detach()
            embed.requires_grad = True

            outputs = self.target.predict(embed)
        
            loss = self.target.get_loss(outputs)
            loss.backward()

            result = self.target.get_result(outputs)
            grads = embed.grad

            self.update_bytes(bytez, bytez_len, grads, result, target_idxs)
        
        embed = self.embedding(bytez).detach()
        final_outputs = self.target.predict(embed)
        final_result = self.target.get_result(final_outputs)
        return final_result

    def do_header_attack(self, bytez, bytez_len):
        target_idxs = []
        for idx in range(len(bytez)):
            target_idxs.append(torch.tensor(list(range(2, 0x3c)), dtype=torch.long))    
            
        for i in range(self.iter_num):
            embed = self.embedding(bytez).detach()
            embed.requires_grad = True

            outputs = self.target.predict(embed)

            loss = self.target.get_loss(outputs)
            loss.backward()
            grads = embed.grad
            
            result = self.target.get_result(outputs)
            self.update_bytes(bytez, bytez_len, grads, result, target_idxs)
        
        embed = self.embedding(bytez).detach()
        final_outputs = self.target.predict(embed)
        return self.target.get_result(final_outputs)

import torch

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FGMAttack:
    def __init__(self, target, embedding):
        self.target = target
        self.embedding = embedding

        self.M = embedding(torch.arange(1, 257).to(device))

    def embedding_Mapping(self, embed_x):
        shortest_bytes = torch.argmin(torch.cdist(embed_x, self.M, p = 2), dim=-1)+1
        return shortest_bytes

    def do_append_attack(self, bytez, bytez_len, padding_len, eps):
        embed = self.embedding(bytez).detach()
        embed.requires_grad = True

        outputs = self.target.predict(embed)
        labels = torch.zeros_like(outputs).to(device)
        loss = self.target.loss_function(outputs, labels)
        loss.backward()

        grads = embed.grad
        for idx in range(len(bytez)):
            origin_len = int(bytez_len[idx])
            padding = bytez[idx, origin_len:origin_len+padding_len]
            padding_grad = grads[idx, origin_len:origin_len+padding_len]

            z = self.embedding(padding)
            z = z - eps * padding_grad.detach().sign()
            bytez[idx, origin_len:origin_len+padding_len] = self.embedding_Mapping(z)
       
        embed = self.embedding(bytez).detach()
        final_outputs = self.target.predict(embed)
        return self.target.get_result(final_outputs)

    def do_slack_attack(self, bytez, target_idxs, eps):
        embed = self.embedding(bytez).detach()
        embed.requires_grad = True

        outputs = self.target.predict(embed)
        labels = torch.zeros_like(outputs).to(device)
        loss = self.target.loss_function(outputs, labels)
        loss.backward()

        grads = embed.grad
        for idx in range(len(bytez)):
            bytez_index = target_idxs[idx]
            target_bytes = bytez[idx, bytez_index]
            target_grad = grads[idx, bytez_index]

            z = self.embedding(target_bytes)
            z = z - eps * target_grad.detach().sign()
            bytez[idx, bytez_index] = self.embedding_Mapping(z)
       
        embed = self.embedding(bytez).detach()
        final_outputs = self.target.predict(embed)
        return self.target.get_result(final_outputs)

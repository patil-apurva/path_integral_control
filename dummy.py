import torch 
a=torch.tensor([[1., -1.], [1., -1.]])
print(a)

a = a.to(device='cuda')

b=1
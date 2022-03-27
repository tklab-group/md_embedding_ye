import torch
import numpy


x = torch.arange(1., 6.)
values, indices = torch.topk(x, 3)
print(x)
print('indices', indices)
indices.to('cpu')

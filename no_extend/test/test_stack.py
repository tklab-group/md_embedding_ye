import torch
import numpy
from torch.nn.parameter import Parameter
from torch.nn import init
import sys
sys.path.append('../../')

# idx = [0, 1, 2, 3, 4]
num_embeddings = 3
embedding_dim = 4
add_n = 2
factory_kwargs = {'device': None, 'dtype': None}
weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
init.normal_(weight)
print('weight', weight)

new_param = Parameter(torch.empty((add_n, embedding_dim), **factory_kwargs))
init.normal_(new_param)
print('new_param', new_param)
new_weight = torch.cat([weight, new_param], dim=0)
print('new_weight', new_weight)
# weight.data.copy_(new_weight)
weight = new_weight

print('weight', weight)

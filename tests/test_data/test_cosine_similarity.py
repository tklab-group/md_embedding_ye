import sys
sys.path.append('../../')

import torch

list1 = [0.1, 0.2]
list1 = torch.Tensor(list1)
list1 = list1.unsqueeze(0).float()

list2 = [0.2, 0.5]
list2 = torch.Tensor(list2)
list2 = list2.unsqueeze(0).float()

list3 = [
    [0.1, 0.2],
    [0.2, 0.5]
]
list3 = torch.Tensor(list3)
print(list1, list2, list3)

cs = torch.cosine_similarity(list1, list2)
print(cs)

cs2 = torch.cosine_similarity(list3, list1)
print(cs2)

# test cat and stack
list4 = torch.Tensor([
    [0.4, 0.6]
])
list5 = torch.cat((list4, list1), dim=0)
print('list5', list5)
cs3 = torch.cosine_similarity(list5, list1)
print(cs3)

list6 = torch.cat((torch.Tensor([]), list1), dim=0)
print(list6)


import sys
sys.path.append('../../')

import torch

origin_score = torch.Tensor([4, 6, 1, 2])
other_score = torch.Tensor([2])
new_score = torch.cat((origin_score, other_score), dim=0)
print(new_score)

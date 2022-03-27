import sys
sys.path.append('../../')
import torch
import numpy
from torch.nn import functional as F

# dim = 3, batch_size = 2
pos_embedding = torch.Tensor([
    [[0.1, 0.2, 0.3]],
    [[0.4, 0.5, 0.6]]
])
print(pos_embedding, pos_embedding.shape)
hidden_state = torch.Tensor([
    [0.7, 0.8, 0.9],
    [0.10, 0.11, 0.12]
])
print(hidden_state, hidden_state.shape)
hidden_state = hidden_state.unsqueeze(2)
print(hidden_state, hidden_state.shape)
pos_dot = torch.bmm(pos_embedding, hidden_state)
print(pos_dot.shape, pos_dot)
print(1*7+2*8+3*9, 4*10+5*11+6*12)

log_pos = F.logsigmoid(pos_dot)
print(log_pos.shape, log_pos)
print(log_pos.sum(1))
log_pos_sum = log_pos.sum(1)
# negative sampling num=3
neg_embedding = torch.Tensor([
    [[0.4, 0.3, 0.1], [0.4, 0.3, 0.1], [0.4, 0.3, 0.1]],
    [[0.3, 0.5, 0.2], [0.3, 0.5, 0.2], [0.3, 0.5, 0.2]]
])
neg_dot = torch.bmm(neg_embedding, -hidden_state)
log_neg = F.logsigmoid(neg_dot)
print('neg', log_neg.shape, log_neg)
log_neg_sum = log_neg.sum(1)
print('log_neg_sum', log_neg_sum)

print('+', log_pos_sum, log_neg_sum)
loss = log_pos_sum + log_neg_sum
print('loss', loss)

result = -loss.mean()
print('result', result)

# sigmoid_value = torch.sigmoid(pos_dot)
# print(sigmoid_value.shape, sigmoid_value)


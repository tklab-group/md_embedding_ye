import sys
sys.path.append('../../')
from model.eval import merge_top_k

print(1)

prob_list_1 = [0.9, 0.1, 0.001]
aq_list_1 = ['a1', 'a2', 'a3']
prob_list_2 = []
aq_list_2 = []
k = 10

result = merge_top_k(prob_list_1, aq_list_1, prob_list_2, aq_list_2, k)
print(result)

l = ['a']
print(l)


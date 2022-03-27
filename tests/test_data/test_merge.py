import sys

sys.path.append('../../')
from model.mix_eval import merge

model_word_list = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
co_change_word_list = ['c1', 'c2', 'c3', 'm1']
k = 10
print(merge(model_word_list, co_change_word_list, k))

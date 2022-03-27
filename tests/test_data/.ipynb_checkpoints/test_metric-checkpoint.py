import sys
sys.path.append('../../')
from model.metric import Metric


k = 3
metric = Metric(k)
# commit_th, rank_i_c, rec_i_c_len
metric.eval_with_commit(0, 3, 5)
metric.eval_with_commit(0, 0, 5)
metric.eval_with_commit(1, 0, 5)
# metric.eval_with_commit(2, 1, 5)
result1 = metric.summary()
print(result1)
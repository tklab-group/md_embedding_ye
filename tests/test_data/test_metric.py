import sys
sys.path.append('../../')
from model.metric import Metric


k = 3
metric = Metric()
test_data = [
    {
        'commit_th': 0,
        'len_com': 3,
        'no_new_len_com': 2,
        'list': [
            {
                'rank': 1,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 3,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': False
            },
        ]
    },
    {
        'commit_th': 1,
        'len_com': 3,
        'no_new_len_com': 3,
        'list': [
            {
                'rank': 1,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 3,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 5,
                'len_rec': 5,
                'is_target_in_training': True
            },
        ]
    },
    {
        'commit_th': 2,
        'len_com': 3,
        'no_new_len_com': 3,
        'list': [
            {
                'rank': 1,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': True
            },
        ]
    },
    {
        'commit_th': 3,
        'len_com': 3,
        'no_new_len_com': 0,
        'list': [
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': False
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': False
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': False
            },
        ]
    },
    {
        'commit_th': 4,
        'len_com': 3,
        'no_new_len_com': 3,
        'list': [
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 5,
                'is_target_in_training': True
            },
        ]
    },
    {
        'commit_th': 5,
        'len_com': 3,
        'no_new_len_com': 3,
        'list': [
            {
                'rank': 0,
                'len_rec': 0,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 0,
                'is_target_in_training': True
            },
            {
                'rank': 0,
                'len_rec': 0,
                'is_target_in_training': True
            },
        ]
    },
]
for i in range(len(test_data)):
    item = test_data[i]
    commit_th = item['commit_th']
    len_com = item['len_com']
    no_new_len_com = item['no_new_len_com']
    for j in range(len(item['list'])):
        sub_item = item['list'][j]
        rank = sub_item['rank']
        len_rec = sub_item['len_rec']
        is_target_in_training = sub_item['is_target_in_training']
        print(commit_th, rank, len_rec, is_target_in_training)
        metric.eval_with_commit(commit_th, rank, len_rec, is_target_in_training)
# commit_th, rank_i_c, rec_i_c_len
# metric.eval_with_commit(0, 3, 5, True)
# metric.eval_with_commit(0, 0, 5, True)
# metric.eval_with_commit(1, 0, 5, True)
# metric.eval_with_commit(2, 1, 5)
print('consider new file', metric.summary())
print('no consider new file', metric.summary(False))
micro_recall, macro_recall = metric.summary()
k = 5
is_consider_new_file = True
print('k %d | new file %d | micro recall %.2f | macro recall %.2f'
              % (k, is_consider_new_file, micro_recall, macro_recall))

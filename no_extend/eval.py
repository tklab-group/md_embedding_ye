import sys
sys.path.append('..')
import pickle
import numpy as np
import heapq
from metric import Metric
import time
import torch
from config.config_default import get_config


class Evaluation:
    def __init__(self, model, data_loader):
        self.in_embed = model.in_embed
        self.out_embed = model.out_embed

        self.validate_data = data_loader.validate_data
        self.model = model
        self.data_loader = data_loader
        self.vocab = data_loader.vocab

        # 所有的md数据相关的
        self.all_word_to_id = data_loader.all_word_to_id
        self.all_id_to_word = data_loader.all_id_to_word

        # 词库
        self.word_to_id = self.vocab.word_to_id
        self.id_to_word = self.vocab.id_to_word

        # md数据和词库相关的
        self.md_id_to_word_id = self.vocab.md_id_to_word_id
        self.word_id_to_md_id = self.vocab.word_id_to_md_id

        config_all = get_config()
        self.config = config_all['dataset']
        # <PADDING>
        self.padding_word = self.config['padding_word']
        self.padding_id = self.config['padding_id']

    # 这里的contexts_ids都是module data里面的id，要先检查是不是在词库里面
    # 同时，这里返回的id也是module data里面的id
    # 可以记录一下每次预测的情况，比如OOV的情况，和实际预测的结果是怎么样的，用于debug
    def predict(self, contexts_ids, k):
        word_ids = []
        oov_count = 0
        for i in range(len(contexts_ids)):
            contexts_id = contexts_ids[i]
            if contexts_id in self.md_id_to_word_id:
                word_id = self.md_id_to_word_id[contexts_id]
                word_ids.append(word_id)
            # if contexts_id in self.id_to_word:
            #     word_ids.append(contexts_id)
            else:
                oov_count += 1
        if len(word_ids) == 0:
            # print('all oov', contexts_ids)
            return oov_count, []
        idx = torch.tensor(word_ids, dtype=torch.long, device=self.model.device)
        # print('idx', idx.shape)
        contexts_embedding = self.in_embed(idx)
        # print('contexts_embedding', contexts_embedding.shape)
        hidden_state = torch.sum(contexts_embedding, dim=0)
        # print('hidden_state', hidden_state.shape)
        score = self.out_embed(hidden_state)
        # print('score', score)
        softmax = torch.softmax(score, dim=0)
        predict_ids = heapq.nlargest(k, range(len(softmax)), softmax.__getitem__)
        # print('softmax', softmax.shape)
        # predict_ids = heapq.nlargest(k, range(len(score)), score.__getitem__)
        # print('predict_ids', predict_ids)
        # print('score', score)
        # print('result_ids', result_ids)
        # print('test', 'src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)' in self.word_to_id)
        # result_ids = []
        # for i in range(len(predict_ids)):
        #     if predict_ids[i] == self.padding_id:
        #         # 先确定md_id里面有没有一样的padding_id,md_id里面最小的也是0，和padding_id冲突，这里返回一个不存在的md_id作为没预测成功的一个flag
        #         print('predicted padding id')
        #     else:
        #         result_ids.append(self.word_id_to_md_id[predict_ids[i]])
        # return result_ids
        return oov_count, predict_ids

    # 如果存在不是词库的数据，应该想办法返回和统计
    def predict_string(self, contexts_str, k):
        contexts_ids = []
        for i in range(len(contexts_str)):
            word = contexts_str[i]
            # word_to_id是词库里面的数据
            if word in self.word_to_id:
                word_id = self.word_to_id[word]
                contexts_ids.append(word_id)
        return self.predict(contexts_ids, k)

    # target md_id
    # aq word_id set including padding_id
    def rank(self, aq, target):
        # target is OOV
        if target not in self.md_id_to_word_id:
            return 0
        for i in range(len(aq)):
            if aq[i] != self.padding_id and self.word_id_to_md_id[aq[i]] == target:
                return i + 1
        return 0

    # target word_id
    # aq word_id set including padding_id
    def rank_train(self, aq, target):
        for i in range(len(aq)):
            if aq[i] != self.padding_id and aq[i] == target:
                return i + 1
        return 0

    def validate(self, k):
        start_time = time.time()
        # leave one out
        contexts = self.validate_data['contexts']
        target = self.validate_data['target']
        # contexts = self.data_loader.train_data['contexts']
        # target = self.data_loader.train_data['target']
        print('total validate data size', len(contexts))
        metric = Metric(k)
        # commit_th 第几个commit
        # rank_i_c 对于缺失的c，进行推荐时，c的rank
        # rec_i_c_len 这次推荐的长度
        commit_th = -1
        rank_i_c = 0
        rec_i_c_len = 0
        # 标记是否要进入下个commit了
        is_next_commit = True
        commit_len_i = 0
        total_contexts_count = 0
        total_oov_count = 0
        commit_th_hit_count = 0
        for i in range(len(contexts)):
            # print(len(contexts[i]))
            if is_next_commit:
                commit_len_i = len(contexts[i]) + 1
                commit_th += 1
                commit_th_hit_count = 0
            # print('commit_th', commit_th, contexts[i], target[i])
            # predicted result of query
            oov_count, aq = self.predict(contexts[i], k)
            total_contexts_count += len(contexts[i])
            total_oov_count += oov_count
            rec_i_c_len = len(aq)
            rank_i_c = self.rank(aq, target[i])
            # rank_i_c = self.rank_train(aq, target[i])
            if rank_i_c > 0:
                commit_th_hit_count += 1
            metric.eval_with_commit(commit_th, rank_i_c, rec_i_c_len)
            elapsed_time = time.time() - start_time
            # if i % 100 == 0:
            #     print('iter %d | time %d[s]'
            #           % (i, elapsed_time))

            commit_len_i -= 1
            if commit_len_i == 0:
                is_next_commit = True
                if commit_th > 0 and commit_th % 1 == 0:
                    # print(contexts[i], target[i], aq, rank_i_c)
                    print('commit_th %d | time %d[s] | tor %.2f | cor %.2f '
                          '| contexts size %d | hit rate %f | (Recall, MRR, F_MRR) =>'
                          % (commit_th, elapsed_time,
                             (100 * total_oov_count / total_contexts_count),
                             100 * oov_count / len(contexts[i]),
                             len(contexts[i]),
                             100 * commit_th_hit_count / (len(contexts[i]) + 1)),
                          metric.summary())
            else:
                is_next_commit = False
        return metric.summary()

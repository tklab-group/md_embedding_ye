import sys

sys.path.append('../')
import pickle
import numpy as np
import heapq
from model.metric import Metric
import time
import torch
from config.config_default import get_config
from data.mode_enum import Mode
from data.data_loader import DataLoader
from statistics import mean, median, stdev, variance
from co_change.handle_target import HandleTarget
from data.util import get_co_change
from model.eval import Evaluation as ModelEvaluation, get_top_k
from co_change.eval import Evaluation as CoChangeEvaluation
from common.util import leave_one_out


class MixEvaluationSubword:
    def __init__(self,
                 model_normal,
                 data_loader_normal: DataLoader,
                 model_subword,
                 data_loader_subword: DataLoader
                 ):
        self.model_normal = model_normal
        self.data_loader_normal = data_loader_normal
        self.model_subword = model_subword
        self.data_loader_subword = data_loader_subword

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.padding_word = config_all['dataset']['padding_word']

        self.k_list = [1, 5, 10, 15, 20]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()

        self.out_embedding_index_to_word_normal = data_loader_normal.embeddingIndexMapped.out_embedding_index_to_word
        self.out_embedding_index_to_word_subword = data_loader_subword.embeddingIndexMapped.out_embedding_index_to_word

        self.modelEvaluationNormal = ModelEvaluation(
            model_normal,
            data_loader_normal,
            data_loader_normal.mode,
            False,
            False
        )

        self.modelEvaluationSubword = ModelEvaluation(
            model_subword,
            data_loader_subword,
            data_loader_subword.mode,
            False,
            False
        )

        # record
        self.rank_prob_list = []
        self.rank_prob_subword_list = []
        self.subword_count = 0
        self.no_predict_count = 0

    def validate_with_transaction(self, transaction, commit_th, commit_hash=None):
        # i = commit_th
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            result[k] = {
                'commit_th_list': [],
                'rank_i_c_list': [],
                'rec_i_c_len_list': [],
                'is_target_in_train_list': []
            }
        # pair_list' data is embedding index
        pair_list, max_length, max_sub_length = \
            self.data_loader_normal.contextsTargetBuilder.get_contexts_target(
                transaction,
                self.modelEvaluationNormal.md_id_to_word,
                False,
                commit_hash)
        pair_list_subword, max_length_subword, max_sub_length_subword = \
            self.data_loader_subword.contextsTargetBuilder.get_contexts_target(
                transaction,
                self.modelEvaluationSubword.md_id_to_word,
                False,
                commit_hash)
        pair_result = []
        # in one commit
        for j in range(len(pair_list)):
            pair = pair_list[j]
            contexts = pair['contexts']
            target = pair['target']
            target_word = pair['target_word']
            # target_embedding_index = pair['target_embedding_index']
            # self.debug_target(target, target_embedding_index)
            # if contexts is empty, no feedback
            if len(contexts) > 0:
                predict_result = self.modelEvaluationNormal.predict(contexts)
                for i in range(len(self.k_list)):
                    k = self.k_list[i]
                    probability = predict_result[k]['prob_list']
                    aq = predict_result[k]['aq']
                    rec_i_c_len = len(aq)
                    rank_i_c, target_prob, is_hit_new_file = self.modelEvaluationNormal.rank(aq, probability, target)
                    result[k]['rank_i_c_list'].append(rank_i_c)
                    result[k]['rec_i_c_len_list'].append(rec_i_c_len)
                    self.rank_prob_list.append({
                        'k': k,
                        'prob': target_prob,
                        'rank': rank_i_c
                    })
                pair['top100_aq'] = []
                for i in range(len(predict_result[100]['aq'])):
                    cur_val = int(predict_result[100]['aq'][i].to('cpu').numpy())
                    pair['top100_aq'].append(cur_val)
                pair['top100_prob_list'] = []
                for i in range(len(predict_result[100]['prob_list'])):
                    cur_val = float(predict_result[100]['prob_list'][i].to('cpu').detach().numpy())
                    pair['top100_prob_list'].append(cur_val)
            else:
                # subword model try to predict
                pair_subword = pair_list_subword[j]
                contexts_subword = pair_subword['contexts']
                self.subword_count += 1
                if len(contexts_subword) > 0:
                    # ---------------------------------------------
                    predict_result = self.modelEvaluationSubword.predict(contexts_subword)
                    for i in range(len(self.k_list)):
                        k = self.k_list[i]
                        probability = predict_result[k]['prob_list']
                        aq = predict_result[k]['aq']
                        rec_i_c_len = len(aq)
                        rank_i_c, target_prob, is_hit_new_file = self.modelEvaluationSubword.rank(
                            aq,
                            probability,
                            target)
                        result[k]['rank_i_c_list'].append(rank_i_c)
                        result[k]['rec_i_c_len_list'].append(rec_i_c_len)
                        self.rank_prob_list.append({
                            'k': k,
                            'prob': target_prob,
                            'rank': rank_i_c
                        })
                        self.rank_prob_subword_list.append({
                            'k': k,
                            'prob': target_prob,
                            'rank': rank_i_c
                        })
                    pair['top100_aq'] = []
                    for i in range(len(predict_result[100]['aq'])):
                        cur_val = int(predict_result[100]['aq'][i].to('cpu').numpy())
                        pair['top100_aq'].append(cur_val)
                    pair['top100_prob_list'] = []
                    for i in range(len(predict_result[100]['prob_list'])):
                        cur_val = float(predict_result[100]['prob_list'][i].to('cpu').detach().numpy())
                        pair['top100_prob_list'].append(cur_val)
                    # ---------------------------------------------
                else:
                    self.no_predict_count += 1
                    for i in range(len(self.k_list)):
                        k = self.k_list[i]
                        result[k]['rank_i_c_list'].append(0)
                        result[k]['rec_i_c_len_list'].append(0)
                    pair['top100_aq'] = []
                    pair['top100_prob_list'] = []
            pair_result.append(pair)
            for i in range(len(self.k_list)):
                k = self.k_list[i]
                result[k]['commit_th_list'].append(commit_th)
                if not target == -1:
                    result[k]['is_target_in_train_list'].append(True)
                else:
                    result[k]['is_target_in_train_list'].append(False)
        return result, pair_result

    def validate(self):
        for i in range(len(self.modelEvaluationNormal.validate_data)):
            transaction = self.modelEvaluationNormal.validate_data[i]
            transaction_result, pair_result = self.validate_with_transaction(
                                    transaction,
                                    i,
                                    self.modelEvaluationNormal.validate_data_commit_hash_list[i])
            for ii in range(len(self.k_list)):
                k = self.k_list[ii]
                for iii in range(len(transaction)):
                    self.metric_list[k].eval_with_commit(
                        transaction_result[k]['commit_th_list'][iii],
                        transaction_result[k]['rank_i_c_list'][iii],
                        transaction_result[k]['rec_i_c_len_list'][iii],
                        transaction_result[k]['is_target_in_train_list'][iii],
                    )
        self.debug_rank_prob_subword()
        return self.metric_list

    def debug_rank_prob_subword(self):
        average = 1 / len(self.data_loader_normal.embeddingIndexMapped.word_to_out_embedding_index)
        print('debug_rank_prob, average', average)
        print('subword total, no predict', self.subword_count, self.no_predict_count)
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            print(k, '-' * 16)
            over_30_count = 0
            over_50_count = 0
            over_80_count = 0
            over_average_count = 0
            lower_average_count = 0
            total = 0
            hit_total = 0
            for j in range(len(self.rank_prob_subword_list)):
                item = self.rank_prob_subword_list[j]
                item_k = item['k']
                prob = item['prob']
                rank = item['rank']
                if k == item_k:
                    total += 1
                    if rank > 0:
                        hit_total += 1
                        if prob >= 0.8:
                            over_80_count += 1
                            over_50_count += 1
                            over_30_count += 1
                        elif prob >= 0.5:
                            over_30_count += 1
                            over_50_count += 1
                        elif prob >= 0.3:
                            over_30_count += 1
                        if prob >= average:
                            over_average_count += 1
                        if prob < average:
                            lower_average_count += 1
            print('over_30: %d | over_50 %d | over_80 %d | over_average %d | lower_average %d | total %d | hit_total %d'
                  % (over_30_count, over_50_count, over_80_count, over_average_count, lower_average_count, total,
                     hit_total))


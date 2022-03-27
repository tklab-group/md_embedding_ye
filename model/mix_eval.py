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


def rank(word_list, target_word):
    for i in range(len(word_list)):
        if word_list[i] == target_word:
            return i + 1
    return 0


def merge(model_word_list, co_change_word_list, k, contribution_rate=0.5):
    # contribution_rateは共変更の推薦結果の貢献割合
    co_change_k = int(k * contribution_rate)

    if len(co_change_word_list) <= co_change_k:
        co_change_part = co_change_word_list
    else:
        co_change_part = co_change_word_list[0: co_change_k]

    co_change_part_set = set(co_change_part)
    count = 0

    model_k = k - len(co_change_part)
    model_part = []
    for i in range(len(model_word_list)):
        if count >= model_k:
            break
        word = model_word_list[i]
        if word not in co_change_part_set:
            model_part.append(word)
            count += 1

    result = co_change_part
    for i in range(len(model_part)):
        result.append(model_part[i])
    return result


class MixEvaluation:
    def __init__(self,
                 model,
                 data_loader: DataLoader,
                 mode=Mode.NORMAL,
                 is_negative_sampling=False,
                 is_cosine_similarity_predict=False,
                 is_fix_transaction=True,
                 contribution_rate=0.5):
        self.model = model
        self.data_loader = data_loader
        self.mode = mode
        self.is_negative_sampling = is_negative_sampling
        self.is_cosine_similarity_predict = is_cosine_similarity_predict
        self.is_fix_transaction = is_fix_transaction
        self.contribution_rate = contribution_rate

        self.git_name = data_loader.git_name
        self.expected_validate_length = data_loader.expected_validate_length
        self.most_recent = data_loader.most_recent

        self.train_id_to_word = data_loader.idMapped.train_id_to_word
        self.out_embedding_index_to_word = data_loader.embeddingIndexMapped.out_embedding_index_to_word

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.padding_word = config_all['dataset']['padding_word']

        if is_fix_transaction:
            git_name_result = self.git_name + '_true'
        else:
            git_name_result = self.git_name + '_false'
        if self.most_recent > 0:
            git_name_result += '_' + str(self.most_recent)
        # print('git_name_result', git_name_result)
        co_change_list = get_co_change(git_name_result)
        handleTarget = HandleTarget(
            self.git_name,
            self.expected_validate_length,
            self.most_recent)
        self.co_change_list = handleTarget.filter(co_change_list, is_fix_transaction)
        # self.co_change_list = co_change_list

        self.coChangeEvaluation = CoChangeEvaluation(self.co_change_list)
        self.modelEvaluation = ModelEvaluation(
            model,
            data_loader,
            mode,
            is_negative_sampling,
            is_cosine_similarity_predict
        )

        self.k_list = [2, 10, 20]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()

        # self.debug()

    def to_word_from_md_id(self, md_id_list):
        result = []
        for i in range(len(md_id_list)):
            md_id = md_id_list[i]
            word = self.train_id_to_word[md_id]
            result.append(word)
        return result

    def to_word_from_embedding_index(self, embedding_index_list):
        # print('embedding_index_list', embedding_index_list)
        result = []
        for i in range(len(embedding_index_list)):
            embedding_index = int(embedding_index_list[i].to('cpu').numpy())
            # print(type(embedding_index))
            word = self.out_embedding_index_to_word[embedding_index]
            result.append(word)
        return result

    def validate_with_transaction(self,
                                  transaction,
                                  co_change_pair_list,
                                  commit_th,
                                  is_predict_new_file,
                                  commit_hash=None):
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
        # pair_list is embedding index
        pair_list, max_length, max_sub_length = \
            self.modelEvaluation.contextsTargetBuilder.get_contexts_target(
                transaction,
                self.modelEvaluation.md_id_to_word,
                False,
                commit_hash)
        # in one commit
        for j in range(len(pair_list)):
            pair = pair_list[j]
            contexts = pair['contexts']
            target = pair['target']
            target_word = pair['target_word']
            target_embedding_index = pair['target_embedding_index']
            if len(contexts) > 0:
                if not self.is_cosine_similarity_predict:
                    # torch.softmax embedding index
                    predict_result = self.modelEvaluation.predict_copy(contexts)
                else:
                    # torch.cosine_similarity embedding index
                    predict_result = self.modelEvaluation.predict_v2_copy(contexts, target_word, is_predict_new_file)
                # print('predict_result', len(predict_result))
                # Contextと重複したものを削除
                max_k = min(300, len(predict_result) - len(contexts))
                predict_result_prob, predict_result_aq = get_top_k(max_k, contexts, predict_result)
                predict_word_list = self.to_word_from_embedding_index(predict_result_aq)

                # max top-300, md id list
                co_change_result = co_change_pair_list[j]['topKList']
                co_change_word_list = self.to_word_from_md_id(co_change_result)

                for i in range(len(self.k_list)):
                    k = self.k_list[i]
                    merge_list = merge(predict_word_list, co_change_word_list, k, self.contribution_rate)
                    rec_i_c_len = len(merge_list)
                    rank_i_c = rank(merge_list, target_word)
                    result[k]['rank_i_c_list'].append(rank_i_c)
                    result[k]['rec_i_c_len_list'].append(rec_i_c_len)
            else:
                for i in range(len(self.k_list)):
                    k = self.k_list[i]
                    result[k]['rank_i_c_list'].append(0)
                    result[k]['rec_i_c_len_list'].append(0)
            for i in range(len(self.k_list)):
                k = self.k_list[i]
                result[k]['commit_th_list'].append(commit_th)
                if not target == -1:
                    result[k]['is_target_in_train_list'].append(True)
                else:
                    result[k]['is_target_in_train_list'].append(False)

        return result

    def validate(self, is_predict_new_file=True):
        for i in range(len(self.modelEvaluation.validate_data)):
            transaction = self.modelEvaluation.validate_data[i]
            co_change_pair_list = self.co_change_list[i]['list']
            transaction_result = self.validate_with_transaction(
                                    transaction,
                                    co_change_pair_list,
                                    i,
                                    is_predict_new_file,
                                    self.modelEvaluation.validate_data_commit_hash_list[i])
            for ii in range(len(self.k_list)):
                k = self.k_list[ii]
                for iii in range(len(transaction)):
                    self.metric_list[k].eval_with_commit(
                        transaction_result[k]['commit_th_list'][iii],
                        transaction_result[k]['rank_i_c_list'][iii],
                        transaction_result[k]['rec_i_c_len_list'][iii],
                        transaction_result[k]['is_target_in_train_list'][iii],
                    )
        return self.metric_list

    def debug(self):
        print(len(self.modelEvaluation.validate_data), len(self.co_change_list))
        for i in range(len(self.modelEvaluation.validate_data)):
            transaction = self.modelEvaluation.validate_data[i]
            pair_list = leave_one_out(transaction)
            co_change_pair_list = self.co_change_list[i]['list']
            # print(transaction, co_change_pair_list)
            for ii in range(len(pair_list)):
                pair = pair_list[ii]
                contexts = pair['contexts']
                target = pair['target']

                co_pair = co_change_pair_list[ii]
                co_contexts = co_pair['contexts']
                co_target = co_pair['target']

                # print('model', contexts, target)
                # print('co change', co_contexts, co_target)
                # print()

                # check
                if len(contexts) != len(co_contexts):
                    print('error')
                    print('m', contexts, target)
                    print('c', co_contexts, co_target)
                    print()

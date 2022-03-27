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
from data.util import get_co_change
from model.eval import Evaluation, get_top_k
from common.util import leave_one_out, get_file_level_info


def rank(word_list, prob_list, target_word):
    for i in range(len(word_list)):
        if word_list[i] == target_word:
            return i + 1, prob_list[i]
    return 0, 0


def ranking_filter_with_file(predict_word_list,
                             prob_list,
                             predict_file_list,
                             max_k
                             ):
    file_set = set(predict_file_list)
    result = []
    result_set = set()
    result_prob_list = []
    count = 0
    for i in range(len(predict_word_list)):
        if count >= max_k:
            break
        word = predict_word_list[i]
        file = get_file_level_info(word)
        if file in file_set:
            result.append(word)
            result_prob_list.append(prob_list[i])
            result_set.add(word)
            count += 1
    rest_k = max_k - count
    rest_count = 0
    for i in range(len(predict_word_list)):
        if rest_count >= rest_k:
            break
        word = predict_word_list[i]
        if word not in result_set:
            result.append(word)
            result_prob_list.append(prob_list[i])
            result_set.add(word)
            rest_count += 1
    return result, result_prob_list


def ranking(predict_word_list,
            predict_word_prob_list,
            predict_file_list,
            predict_file_prob_list,
            prob_threshold,
            prob_threshold_file_level):
    # return predict_word_list[0:k]
    result = []
    result_prob_list = []
    result_set = set()
    file_bucket = {}
    other_file_token = 'OTHER_FILE'
    # build file level bucket
    for i in range(len(predict_file_list)):
        file = predict_file_list[i]
        if predict_file_prob_list[i] <= prob_threshold_file_level:
            break
        if file not in file_bucket:
            file_bucket[file] = []
        for ii in range(len(predict_word_list)):
            word = predict_word_list[ii]
            if predict_word_prob_list[ii] <= prob_threshold:
                break
            if word not in result_set and file == get_file_level_info(word):
                file_bucket[file].append({
                    'word': word,
                    'prob': predict_word_prob_list[ii]
                })
                result_set.add(word)
    # put not match file level module data to other_file_token bucket
    if len(result_set) < len(predict_word_list):
        file_bucket[other_file_token] = []
        for i in range(len(predict_word_list)):
            word = predict_word_list[i]
            if predict_word_prob_list[i] <= prob_threshold:
                break
            if word not in result_set:
                file_bucket[other_file_token].append({
                    'word': word,
                    'prob': predict_word_prob_list[i]
                })
                result_set.add(word)
    # arrange bucket to list and return this list as result
    for i in range(len(predict_file_list)):
        file = predict_file_list[i]
        if predict_file_prob_list[i] <= prob_threshold_file_level:
            break
        for ii in range(len(file_bucket[file])):
            result.append(file_bucket[file][ii]['word'])
            result_prob_list.append(file_bucket[file][ii]['prob'])
    if other_file_token in file_bucket:
        for i in range(len(file_bucket[other_file_token])):
            result.append(file_bucket[other_file_token][i]['word'])
            result_prob_list.append(file_bucket[other_file_token][i]['prob'])
    return result, result_prob_list


def debug_ranking(predict_word_list,
                  predict_word_prob_list,
                  predict_file_list,
                  predict_file_prob_list,
                  target_word,
                  target,
                  prob_threshold,
                  prob_threshold_file_level
                  ):
    before_rank = rank(predict_word_list, target_word)
    if before_rank != 0:
        for i in range(len(predict_word_list)):
            if target_word == predict_word_list[i]:
                prob = predict_word_prob_list[i].to('cpu').detach().numpy() * 100
                break
    else:
        prob = 0
    first_prob = predict_word_prob_list[0].to('cpu').detach().numpy() * 100
    ranking_result = ranking(
        predict_word_list,
        predict_word_prob_list,
        predict_file_list,
        predict_file_prob_list,
        prob_threshold,
        prob_threshold_file_level
    )
    after_rank = rank(ranking_result, target_word)

    if target != -1 and before_rank <= 20:
        print('rank change', before_rank, after_rank, before_rank - after_rank, prob, first_prob)
    return before_rank, after_rank, before_rank - after_rank


class RankingEvaluation:
    def __init__(self,
                 model,
                 data_loader: DataLoader,
                 model_file_level,
                 data_loader_file_level: DataLoader,
                 pre_ranking_top_k=1000,
                 pre_ranking_file_level_top_k=20,
                 mode=Mode.NORMAL,
                 is_multi=False,
                 ):
        self.model = model
        self.data_loader = data_loader
        self.model_file_level = model_file_level
        self.data_loader_file_level = data_loader_file_level
        self.pre_ranking_top_k = pre_ranking_top_k
        self.pre_ranking_file_level_top_k = pre_ranking_file_level_top_k
        self.mode = mode
        self.is_multi = is_multi
        print('RankingEvaluation param', pre_ranking_top_k, pre_ranking_file_level_top_k, is_multi)

        # method level as same as file level
        self.md_id_to_word = self.data_loader.idMapped.all_id_to_word
        self.validate_data = self.data_loader.validate_data
        self.validate_data_commit_hash_list = self.data_loader.dataDivider.validate_data_commit_hash_list

        self.embeddingIndexMapped = self.data_loader.embeddingIndexMapped
        self.embeddingIndexMappedFileLevel = self.data_loader_file_level.embeddingIndexMapped

        self.contextsTargetBuilder = self.data_loader.contextsTargetBuilder
        self.contextsTargetBuilderFileLevel = self.data_loader_file_level.contextsTargetBuilder

        self.vocab_size = len(self.data_loader.vocab.words)
        self.vocab_file_size = len(self.data_loader_file_level.vocab.files)

        self.evaluation = Evaluation(
            model=model,
            data_loader=data_loader,
            mode=mode,
            is_negative_sampling=False,
            is_cosine_similarity_predict=False,
            is_split=False,
            is_predict_with_softmax_and_merge=False,
            is_predict_with_file_level=False
        )

        self.evaluationFileLevel = Evaluation(
            model=model_file_level,
            data_loader=data_loader_file_level,
            mode=mode,
            is_negative_sampling=False,
            is_cosine_similarity_predict=False,
            is_split=False,
            is_predict_with_softmax_and_merge=False,
            is_predict_with_file_level=True
        )

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.padding_word = config_all['dataset']['padding_word']

        self.k_list = [1, 2, 5, 10, 15, 20, 100]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()

        # record
        self.rank_prob_list = []

    def contexts_to_word_from_embedding_index(self, embedding_index_list):
        # print('embedding_index_list', embedding_index_list)
        result = []
        for i in range(len(embedding_index_list)):
            embedding_index = embedding_index_list[i]
            # print(type(embedding_index))
            word = self.embeddingIndexMapped.in_embedding_index_to_word[embedding_index]
            result.append(word)
        return result

    def contexts_to_word_from_embedding_index_file_level(self, embedding_index_list):
        # print('embedding_index_list', embedding_index_list)
        result = []
        for i in range(len(embedding_index_list)):
            embedding_index = embedding_index_list[i]
            # print(type(embedding_index))
            word = self.embeddingIndexMappedFileLevel.in_embedding_index_to_word[embedding_index]
            result.append(word)
        return result

    def to_word_from_embedding_index(self, embedding_index_list, is_gpu=True):
        # print('embedding_index_list', embedding_index_list)
        result = []
        for i in range(len(embedding_index_list)):
            if is_gpu:
                embedding_index = int(embedding_index_list[i].to('cpu').numpy())
            else:
                embedding_index = embedding_index_list[i]
            # print(type(embedding_index))
            word = self.embeddingIndexMapped.out_embedding_index_to_word[embedding_index]
            result.append(word)
        return result

    def to_word_from_embedding_index_file_level(self, embedding_index_list, is_gpu=True):
        # print('embedding_index_list', embedding_index_list)
        result = []
        for i in range(len(embedding_index_list)):
            if is_gpu:
                embedding_index = int(embedding_index_list[i].to('cpu').numpy())
            else:
                embedding_index = embedding_index_list[i]
            # print(type(embedding_index))
            word = self.embeddingIndexMappedFileLevel.out_embedding_index_to_word[embedding_index]
            result.append(word)
        return result

    def ranking_multi(self, predict_result, predict_file_level_result):
        for i in range(len(predict_result)):
            word = self.embeddingIndexMapped.out_embedding_index_to_word[i]
            file = get_file_level_info(word)
            prob = predict_result[i]
            if file in self.embeddingIndexMappedFileLevel.word_to_out_embedding_index:
                predict_result[i] = \
                    predict_file_level_result[self.embeddingIndexMappedFileLevel.word_to_out_embedding_index[file]] \
                    * prob
        return predict_result

    def validate_with_transaction(self,
                                  transaction,
                                  commit_th,
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
            self.contextsTargetBuilder.get_contexts_target(
                transaction,
                self.md_id_to_word,
                False,
                commit_hash)
        pair_list_file_level, max_length_file_level, max_sub_length_file_level = \
            self.contextsTargetBuilderFileLevel.get_contexts_target(
                transaction,
                self.md_id_to_word,
                False,
                commit_hash)
        if len(pair_list) != len(pair_list_file_level):
            print('pair list is not same!')
        # in one commit
        for j in range(len(pair_list)):
            pair = pair_list[j]
            contexts = pair['contexts']
            target = pair['target']
            target_word = pair['target_word']

            pair_file_level = pair_list_file_level[j]
            contexts_file_level = pair_file_level['contexts']
            # target_file_level = pair_file_level['target']
            # target_word_file_level = pair_file_level['target_word']
            # print(self.contexts_to_word_from_embedding_index(contexts))
            # print(self.contexts_to_word_from_embedding_index_file_level(contexts_file_level))
            if len(contexts) > 0:
                predict_result = self.evaluation.predict_copy(contexts)
                # print('predict_result', predict_result)
                # print(self.pre_ranking_top_k, contexts, predict_result)
                predict_file_level_result = self.evaluationFileLevel.predict_copy(contexts_file_level)
                # print('predict_file_level_result', predict_file_level_result)
                if self.is_multi:
                    multi_result = self.ranking_multi(predict_result, predict_file_level_result)
                    # print('multi_result', multi_result)
                    for i in range(len(self.k_list)):
                        k = self.k_list[i]
                        predict_result_prob, predict_result_aq = get_top_k(
                            k, contexts, multi_result)
                        rec_i_c_len = len(predict_result_aq)
                        rank_i_c, target_prob = rank(predict_result_aq, predict_result_prob, target)
                        result[k]['rank_i_c_list'].append(rank_i_c)
                        result[k]['rec_i_c_len_list'].append(rec_i_c_len)
                else:
                    predict_result_prob, predict_result_aq = get_top_k(self.pre_ranking_top_k, contexts, predict_result)
                    predict_word_list = self.to_word_from_embedding_index(predict_result_aq)

                    predict_result_prob_file_level, predict_result_aq_file_level = \
                        get_top_k(self.pre_ranking_file_level_top_k, [], predict_file_level_result)
                    predict_file_list = self.to_word_from_embedding_index_file_level(predict_result_aq_file_level)

                    # debug_ranking(
                    #     predict_word_list,
                    #     predict_result_prob,
                    #     predict_file_list,
                    #     predict_result_prob_file_level,
                    #     target_word,
                    #     target,
                    #     1 / self.vocab_size,
                    #     1 / self.vocab_file_size
                    # )

                    # 1 / self.vocab_file_size
                    # 1 / self.vocab_size
                    ranking_predict_result, ranking_predict_prob_result = ranking(
                        predict_word_list,
                        predict_result_prob,
                        predict_file_list,
                        predict_result_prob_file_level,
                        0,
                        0
                    )

                    # ranking_predict_result, ranking_predict_prob_result = ranking_filter_with_file(
                    #     predict_word_list,
                    #     predict_result_prob,
                    #     predict_file_list,
                    #     self.k_list[len(self.k_list) - 1]
                    # )

                    for i in range(len(self.k_list)):
                        k = self.k_list[i]
                        if k <= len(ranking_predict_result):
                            ranking_predict_result_top_k = ranking_predict_result[0:k]
                        else:
                            ranking_predict_result_top_k = ranking_predict_result
                        rec_i_c_len = len(ranking_predict_result_top_k)
                        rank_i_c, target_prob = rank(
                            ranking_predict_result_top_k, ranking_predict_prob_result, target_word)
                        result[k]['rank_i_c_list'].append(rank_i_c)
                        result[k]['rec_i_c_len_list'].append(rec_i_c_len)
                        self.rank_prob_list.append({
                            'k': k,
                            'prob': target_prob,
                            'rank': rank_i_c
                        })
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

    def debug_rank_prob(self):
        average = 1 / len(self.embeddingIndexMapped.word_to_out_embedding_index)
        print('debug_rank_prob, average', average)
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
            for j in range(len(self.rank_prob_list)):
                item = self.rank_prob_list[j]
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

    def validate(self):
        start_time = time.time()
        for i in range(len(self.validate_data)):
            transaction = self.validate_data[i]
            transaction_result = self.validate_with_transaction(
                transaction,
                i,
                self.validate_data_commit_hash_list[i])
            for ii in range(len(self.k_list)):
                k = self.k_list[ii]
                for iii in range(len(transaction)):
                    self.metric_list[k].eval_with_commit(
                        transaction_result[k]['commit_th_list'][iii],
                        transaction_result[k]['rank_i_c_list'][iii],
                        transaction_result[k]['rec_i_c_len_list'][iii],
                        transaction_result[k]['is_target_in_train_list'][iii],
                    )
            if i % 10 == 0:
                print('validate:', i, time.time() - start_time)
        self.debug_rank_prob()
        return self.metric_list

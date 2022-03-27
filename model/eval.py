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
from statistics import mean, median, stdev, variance
from common.util import leave_one_out
from data.younger_trace import YoungerTrace
from data.data_divider import DataDivider
from data.id_mapped import IdMapped
from data.delete_record import DeleteRecord


def get_top_k(k, contexts_embedding_index_list, score_list):
    if score_list is None:
        return [], []
    # print('get_top_k', contexts_embedding_index_list, score_list)
    max_k = min(len(score_list), k + len(contexts_embedding_index_list))
    # print(len(score_list), k + len(contexts_embedding_index_list), max_k)
    predict_probability, predict_out_embedding_index_list \
        = torch.topk(score_list, max_k)
    # contextsの部分を除外
    contexts_set = set(contexts_embedding_index_list)
    prob_list = []
    aq = []
    count = 0
    for i in range(len(predict_out_embedding_index_list)):
        if count >= k:
            break
        if not predict_out_embedding_index_list[i] in contexts_set:
            prob_list.append(predict_probability[i])
            aq.append(predict_out_embedding_index_list[i])
            count += 1
    return prob_list, aq


def merge_top_k(prob_list_1, aq_list_1, prob_list_2, aq_list_2, k):
    prob_list = []
    aq = []
    aq_set = set()
    index_1 = 0
    index_2 = 0
    for i in range(k):
        # print(i, index_1, index_2)
        if index_1 >= len(prob_list_1) and index_2 >= len(prob_list_2):
            break
        if index_1 <= len(prob_list_1) - 1:
            cur_prob_1 = prob_list_1[index_1]
        else:
            cur_prob_1 = 0
        if index_2 <= len(prob_list_2) - 1:
            cur_prob_2 = prob_list_2[index_2]
        else:
            cur_prob_2 = 0
        # print('cur prob', cur_prob_1, cur_prob_2)
        if cur_prob_1 >= cur_prob_2:
            # print(aq_list_1[index_1], aq_set)
            if aq_list_1[index_1] not in aq_set:
                prob_list.append(cur_prob_1)
                aq.append(aq_list_1[index_1])
                aq_set.add(aq_list_1[index_1])
                index_1 += 1
        else:
            if aq_list_2[index_2] not in aq_set:
                prob_list.append(cur_prob_2)
                aq.append(aq_list_2[index_2])
                aq_set.add(aq_list_2[index_2])
                index_2 += 1

    return prob_list, aq


class Evaluation:
    def __init__(self,
                 model,
                 data_loader,
                 mode=Mode.NORMAL,
                 is_negative_sampling=False,
                 is_cosine_similarity_predict=False,
                 is_split=False,
                 is_predict_with_softmax_and_merge=True,
                 is_predict_with_file_level=False,
                 is_only_old_file_context=False
                 ):
        # print('is_split', is_split)
        self.model = model
        self.data_loader = data_loader
        self.mode = mode
        self.is_negative_sampling = is_negative_sampling
        self.is_cosine_similarity_predict = is_cosine_similarity_predict
        self.is_split = is_split
        self.is_predict_with_softmax_and_merge = is_predict_with_softmax_and_merge
        self.is_predict_with_file_level = is_predict_with_file_level
        self.is_only_old_file_context = is_only_old_file_context

        self.md_id_to_word = self.data_loader.idMapped.all_id_to_word
        self.train_id_to_word = self.data_loader.idMapped.train_id_to_word
        self.embeddingIndexMapped = self.data_loader.embeddingIndexMapped
        # print('word_to_out_embedding_index', self.embeddingIndexMapped.word_to_out_embedding_index)
        self.validate_data = self.data_loader.validate_data
        self.validate_data_commit_hash_list = self.data_loader.dataDivider.validate_data_commit_hash_list
        self.contextsTargetBuilder = self.data_loader.contextsTargetBuilder

        self.in_embed = model.in_embed
        self.out_embed = model.out_embed

        # self.hit_new_file_count_list = []
        # self.total_new_file_count = 0
        # self.total_hit_new_file_count = 0

        self.alternate_list = []
        self.feature_vector_list = torch.Tensor([])

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.padding_word = config_all['dataset']['padding_word']

        if is_cosine_similarity_predict:
            self.build_feature_vector()

        self.k_list = [1, 2, 5, 10, 15, 20, 100]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()

        # record
        self.rank_prob_list = []

        self.youngerTrace = YoungerTrace(self.data_loader.idMapped.all_id_to_word)

        self.deleteRecord = data_loader.deleteRecord

    def get_feature_vector(self, word):
        word_embedding_index_list, sub_word_embedding_index_list = \
            self.embeddingIndexMapped.get_embedding_index_list_from_context_word(word, word)
        contexts_embedding_index_list = \
            np.concatenate((word_embedding_index_list, sub_word_embedding_index_list), axis=0)
        if len(contexts_embedding_index_list) == 0:
            return None, False
        idx = torch.tensor(contexts_embedding_index_list, dtype=torch.long, device=self.model.device)
        contexts_embedding = self.in_embed(idx)
        # print('self.model', self.model)
        if self.model.is_hidden_state_sum:
            hidden_state = torch.sum(contexts_embedding, dim=0)
        else:
            hidden_state = torch.mean(contexts_embedding, dim=0)
        return hidden_state, True

    def build_feature_vector(self):
        # print('start build feature vector')
        start_time = time.time()
        for word in self.embeddingIndexMapped.word_to_out_embedding_index:
            # print('build_feature_vector', word)
            if word != self.padding_word:
                feature_vector, is_can_encode = self.get_feature_vector(word)
                feature_vector = feature_vector.to('cpu')
                feature_vector = feature_vector.unsqueeze(0).float()
                self.feature_vector_list = torch.cat((
                    self.feature_vector_list,
                    feature_vector), dim=0)
                self.alternate_list.append(word)
        # print('build_feature_vector cost time:', time.time() - start_time)

    def add_feature_vector(self, word):
        if word not in self.embeddingIndexMapped.word_to_out_embedding_index:
            feature_vector, is_can_encode = self.get_feature_vector(word)
            if is_can_encode:
                feature_vector = feature_vector.to('cpu')
                feature_vector = feature_vector.unsqueeze(0).float()
                self.feature_vector_list = torch.cat((
                    self.feature_vector_list,
                    feature_vector), dim=0)
                self.alternate_list.append(word)

    def predict_v2_copy(self, contexts_embedding_index_list, target_word, is_predict_new_file):
        return self._predict_v2(contexts_embedding_index_list, target_word, is_predict_new_file)

    def _predict_v2(self, contexts_embedding_index_list, target_word, is_predict_new_file):
        if len(contexts_embedding_index_list) == 0:
            return [], []

        # if is_predict_new_file:
        #     self.add_feature_vector(target_word)

        idx = torch.tensor(contexts_embedding_index_list, dtype=torch.long, device=self.model.device)
        # print('idx', idx.shape)
        contexts_embedding = self.in_embed(idx)
        # print('contexts_embedding', contexts_embedding.shape)
        if self.model.is_hidden_state_sum:
            contexts_feature_vector = torch.sum(contexts_embedding, dim=0)
        else:
            contexts_feature_vector = torch.mean(contexts_embedding, dim=0)
        contexts_feature_vector = contexts_feature_vector.to('cpu')
        contexts_feature_vector = contexts_feature_vector.unsqueeze(0).float()

        # cosine_similarity_list = []
        # for i in range(len(self.alternate_list)):
        #     feature_vector = self.feature_vector_map[self.alternate_list[i]]
        #     feature_vector = feature_vector.unsqueeze(0).float()
        #     # print('feature_vector', feature_vector)
        #     # print('target_word_feature_vector', target_word_feature_vector)
        #     cosine_similarity = torch.cosine_similarity(contexts_feature_vector,
        #                                                 feature_vector)
        #     cosine_similarity_list.append(cosine_similarity)
        # print('self.feature_vector_list.shape', self.feature_vector_list.shape)
        cosine_similarity_list = torch.cosine_similarity(torch.Tensor(self.feature_vector_list),
                                                         contexts_feature_vector)
        return cosine_similarity_list

    def predict_v2(self, contexts_embedding_index_list, target_word, is_predict_new_file):
        if len(contexts_embedding_index_list) == 0:
            return [], []
        cosine_similarity_list = self._predict_v2(contexts_embedding_index_list, target_word, is_predict_new_file)
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            prob_list, aq = get_top_k(k, contexts_embedding_index_list, cosine_similarity_list)
            result[k] = {
                'prob_list': prob_list,
                'aq': aq
            }
        return result

    def rank_v2(self, aq, probability, target, target_word):
        # print('aq and target', aq, target)
        rank = 0
        prob = 0
        for i in range(len(aq)):
            if self.alternate_list[aq[i]] == target_word:
                rank = i + 1
                prob = probability[i]
                break
        is_hit_new_file = False
        if target == -1 and rank > 0:
            is_hit_new_file = True
        return rank, prob, is_hit_new_file

    def predict_copy(self, contexts_embedding_index_list):
        return self._predict(contexts_embedding_index_list)

    def _predict(self, contexts_embedding_index_list):
        # print('contexts shape predict', contexts_embedding_index_list)
        start_time = time.time()
        if len(contexts_embedding_index_list) == 0:
            return []
        idx = torch.tensor(contexts_embedding_index_list, dtype=torch.long, device=self.model.device)
        # print('idx', idx.shape)
        contexts_embedding = self.in_embed(idx)
        # print('contexts_embedding', contexts_embedding.shape)
        if self.model.is_hidden_state_sum:
            hidden_state = torch.sum(contexts_embedding, dim=0)
        else:
            hidden_state = torch.mean(contexts_embedding, dim=0)
        # print('hidden_state', hidden_state.shape)
        # print('self.out_embed', self.out_embed.weight.shape)
        if not self.is_negative_sampling:
            # if self.is_predict_new_file and len(target_embedding_index_list) > 0:
            #     # this code will get unfair score for new word!
            #     origin_score = self.model.score(torch.Tensor(contexts_embedding_index_list), False)
            #
            #     target_idx = torch.tensor(target_embedding_index_list, dtype=torch.long, device=self.model.device)
            #     target_embedding = self.in_embed(target_idx)
            #     target_feature_vector = torch.mean(target_embedding, dim=0)
            #     target_score = torch.Tensor([torch.dot(hidden_state, target_feature_vector)])
            #
            #     score = torch.cat((origin_score, target_score), dim=0)
            # else:
            #     # [V_size]
            #     score = self.model.score(torch.Tensor(contexts_embedding_index_list), False)
            score = self.model.score(torch.Tensor(contexts_embedding_index_list), False)
        else:
            # [V_size] => [1, V_size]
            hidden_state = torch.unsqueeze(hidden_state, 0)
            # print('hidden_state', hidden_state.shape)
            # print('self.out_embed.weight.t() shape', self.out_embed.weight.t().shape)
            # [1, V_size] => [V_size]
            score = torch.mm(hidden_state, self.out_embed.weight.t())
            # print('score shape', score.shape)
            score = torch.squeeze(score, 0)
        # print('score time', time.time() - start_time)
        # print('score shape', score.shape)
        softmax = torch.softmax(score, dim=0)
        return softmax

    def predict(self,
                contexts_embedding_index_list,
                commit_hash):
        if len(contexts_embedding_index_list) == 0:
            return []
        softmax = self._predict(contexts_embedding_index_list)
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            if self.is_predict_with_file_level:
                prob_list, aq = get_top_k(k, [], softmax)
            else:
                prob_list, aq = self.get_top_k_no_deleted(k, commit_hash, contexts_embedding_index_list, softmax)
                # prob_list, aq = get_top_k(k, contexts_embedding_index_list, softmax)
            result[k] = {
                'prob_list': prob_list,
                'aq': aq
            }
        return result

    def rank(self, aq, probability, target):
        # print('aq and target', aq, target)
        if target == -1:
            return 0, 0, False
        for i in range(len(aq)):
            if aq[i] == target:
                return i + 1, probability[i], False
        return 0, 0, False

    def predict_with_softmax_and_merge(self,
                                       contexts_embedding_index_list,
                                       contexts_subword_embedding_index_list):
        if len(contexts_embedding_index_list) > 0:
            softmax = self._predict(contexts_embedding_index_list)
        else:
            softmax = None
        if len(contexts_subword_embedding_index_list) > 0:
            subword_softmax = self._predict(contexts_subword_embedding_index_list)
        else:
            subword_softmax = None
        softmax_top_100_prob, softmax_top_100_aq = get_top_k(100, contexts_embedding_index_list, softmax)
        subword_softmax_top_100_prob, subword_softmax_top_100_aq = get_top_k(
            100,
            contexts_embedding_index_list,
            subword_softmax)
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            prob_list, aq = merge_top_k(
                softmax_top_100_prob,
                softmax_top_100_aq,
                subword_softmax_top_100_prob,
                subword_softmax_top_100_aq,
                k
            )
            result[k] = {
                'prob_list': prob_list,
                'aq': aq
            }
        return result

    def validate_with_transaction_split(self, transaction, commit_th, commit_hash=None):
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
        pair_list = \
            self.contextsTargetBuilder.get_contexts_target_split_in_validate(transaction,
                                                                             self.md_id_to_word,
                                                                             commit_hash)
        pair_result = []
        # in one commit
        for j in range(len(pair_list)):
            pair = pair_list[j]
            contexts = pair['contexts']
            contexts_subword = pair['contexts_subword']
            target = pair['target']
            target_word = pair['target_word']
            # if contexts is empty, no feedback
            if len(contexts) > 0:
                predict_result = self.predict_with_softmax_and_merge(contexts, contexts_subword)
                for i in range(len(self.k_list)):
                    k = self.k_list[i]
                    probability = predict_result[k]['prob_list']
                    aq = predict_result[k]['aq']
                    rec_i_c_len = len(aq)
                    if not self.is_cosine_similarity_predict:
                        rank_i_c, target_prob, is_hit_new_file = self.rank(aq, probability, target)
                    else:
                        rank_i_c, target_prob, is_hit_new_file = self.rank_v2(aq, probability, target, target_word)
                    result[k]['rank_i_c_list'].append(rank_i_c)
                    result[k]['rec_i_c_len_list'].append(rec_i_c_len)
                pair['top100_aq'] = []
                for i in range(len(predict_result[100]['aq'])):
                    cur_val = int(predict_result[100]['aq'][i].to('cpu').numpy())
                    pair['top100_aq'].append(cur_val)
                pair['top100_prob_list'] = []
                for i in range(len(predict_result[100]['prob_list'])):
                    cur_val = float(predict_result[100]['prob_list'][i].to('cpu').detach().numpy())
                    pair['top100_prob_list'].append(cur_val)
            else:
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

    def validate_with_transaction(self,
                                  transaction,
                                  commit_th,
                                  is_predict_new_file,
                                  commit_hash=None,
                                  is_only_new_file_context=False):
        # i = commit_th
        if self.is_split:
            return self.validate_with_transaction_split(transaction, commit_th, commit_hash)
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            result[k] = {
                'commit_th_list': [],
                'rank_i_c_list': [],
                'rec_i_c_len_list': [],
                'is_target_in_train_list': []
            }
        if is_only_new_file_context:
            # clean cache test
            self.contextsTargetBuilder.context_cache = {}
        # pair_list' data is embedding index
        pair_list, max_length, max_sub_length = \
            self.contextsTargetBuilder.get_contexts_target(
                transaction=transaction,
                id_to_word=self.md_id_to_word,
                is_train=False,
                commit_hash=commit_hash,
                is_only_new_file_context=is_only_new_file_context)
        if self.contextsTargetBuilder.is_contexts_extend:
            # padding
            # print('max_length, sub_max_length', max_length, sub_max_length)
            # print('pair_list', pair_list)
            for i in range(len(pair_list)):
                pair_contexts = pair_list[i]['contexts']
                for j in range(len(pair_contexts)):
                    sub_contexts = pair_contexts[j]
                    # print('sub_contexts len', len(sub_contexts))
                    padding_array = np.arange(max_sub_length - len(sub_contexts), dtype=int)
                    padding_array = np.full_like(padding_array, self.contextsTargetBuilder.padding_id)
                    # merge array and append
                    pair_contexts[j] = np.concatenate((sub_contexts, padding_array), axis=0)
                pair_list[i] = {
                    'contexts': pair_contexts,
                    'target': pair_list[i]['target']
                }
        pair_result = []
        # in one commit
        if len(pair_list) == 0:
            for i in range(len(self.k_list)):
                k = self.k_list[i]
                result[k]['rank_i_c_list'].append(0)
                result[k]['rec_i_c_len_list'].append(0)
        for j in range(len(pair_list)):
            pair = pair_list[j]
            contexts = pair['contexts']
            target = pair['target']
            target_word = pair['target_word']
            # target_embedding_index = pair['target_embedding_index']
            # self.debug_target(target, target_embedding_index)
            # if contexts is empty, no feedback
            if len(contexts) > 0:
                if not self.is_cosine_similarity_predict:
                    predict_result = self.predict(contexts, commit_hash)
                else:
                    predict_result = self.predict_v2(contexts, target_word, is_predict_new_file)
                for i in range(len(self.k_list)):
                    k = self.k_list[i]
                    probability = predict_result[k]['prob_list']
                    aq = predict_result[k]['aq']
                    rec_i_c_len = len(aq)
                    if not self.is_cosine_similarity_predict:
                        rank_i_c, target_prob, is_hit_new_file = self.rank(aq, probability, target)
                    else:
                        rank_i_c, target_prob, is_hit_new_file = self.rank_v2(aq, probability, target, target_word)
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

    def get_top_k_no_deleted(self, k, cur_commit_hash, contexts_embedding_index_list, score_list):
        if score_list is None:
            return [], []
        # print('get_top_k', contexts_embedding_index_list, score_list)
        deleted_set_count = 0
        if cur_commit_hash in self.deleteRecord.commit_hash_to_deleted_set:
            deleted_set_count = len(self.deleteRecord.commit_hash_to_deleted_set[cur_commit_hash])
        max_k = min(len(score_list), k + len(contexts_embedding_index_list) + deleted_set_count)
        # print(len(score_list), k + len(contexts_embedding_index_list), max_k)
        predict_probability, predict_out_embedding_index_list \
            = torch.topk(score_list, max_k)
        # contextsの部分を除外
        contexts_set = set(contexts_embedding_index_list)
        prob_list = []
        aq = []
        count = 0
        for i in range(len(predict_out_embedding_index_list)):
            if count >= k:
                break
            embedding_index = int(predict_out_embedding_index_list[i].to('cpu').numpy())
            # print('embedding_index', embedding_index)
            word = self.embeddingIndexMapped.out_embedding_index_to_word[embedding_index]
            if (not embedding_index in contexts_set) and self.deleteRecord.detect_is_in_vocab(cur_commit_hash, word):
                prob_list.append(predict_probability[i])
                aq.append(predict_out_embedding_index_list[i])
                count += 1
            # else:
            #     if self.deleteRecord.detect_is_in_vocab(cur_commit_hash, word):
            #         print('check delete', cur_commit_hash, word)
        return prob_list, aq

    def debug_target(self, target, target_embedding_index, target_word):
        origin_target_word = target_word
        # print('target_embedding_index', target_embedding_index)
        if target in self.embeddingIndexMapped.in_embedding_index_to_word:
            target_word = self.embeddingIndexMapped.in_embedding_index_to_word[target]
        else:
            target_word = 'unknown'
        target_sub_word_list = []
        for i in range(len(target_embedding_index)):
            target_sub_word = self.embeddingIndexMapped.in_embedding_index_to_word[target_embedding_index[i]]
            target_sub_word_list.append(target_sub_word)
        print('debug_target', target, target_word, origin_target_word, target_embedding_index, target_sub_word_list)

    def debug_predict(self, predict_out_embedding_index_list):
        predict_word_list = []
        for i in range(len(predict_out_embedding_index_list)):
            embedding_index = predict_out_embedding_index_list[i]
            if embedding_index in self.embeddingIndexMapped.out_embedding_index_to_word:
                predict_word_list.append(self.embeddingIndexMapped.out_embedding_index_to_word[embedding_index])
            else:
                predict_word_list.append('new word!')
        print('debug predict', predict_word_list)

    def debug_hit_new_file_count(self):
        return
        # print('hit_new_file_count mean %f | median %f | stdev %f | variance %f'
        #       % (mean(self.hit_new_file_count_list),
        #          median(self.hit_new_file_count_list),
        #          stdev(self.hit_new_file_count_list),
        #          variance(self.hit_new_file_count_list)))
        # if self.total_new_file_count > 0:
        #     print('average hit new file rate', self.total_hit_new_file_count/self.total_new_file_count)

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

    def get_only_old_file(self, transaction):
        result = []
        for i in range(len(transaction)):
            item = transaction[i]
            if item in self.train_id_to_word:
                result.append(item)
        return result

    def validate(self, is_predict_new_file=False, is_only_new_file_context=False):
        start_time = time.time()
        metric = Metric()
        start_time = time.time()
        # print('validate data(md ids)', self.validate_data)
        for i in range(len(self.validate_data)):
            # if i % 10 == 0:
            #     elapsed_time = time.time() - start_time
            #     print('| commit th %d | time %d[s]'
            #           % (i, elapsed_time))
            transaction = self.validate_data[i]
            if self.is_only_old_file_context:
                transaction = self.get_only_old_file(transaction)
            # print('fix', i, self.validate_data_commit_hash_list[i])
            transaction_result, pair_result = self.validate_with_transaction(
                transaction,
                i,
                is_predict_new_file,
                self.validate_data_commit_hash_list[i],
                is_only_new_file_context)
            # print('pair_result', pair_result)
            # # test
            # pair_list = leave_one_out(transaction)
            # print(len(pair_list), len(pair_result))
            # for ii in range(len(pair_result)):
            #     # check target is same
            #     target_word = self.md_id_to_word[pair_list[ii]['target']]
            #     print('is same', target_word == pair_result[ii]['target_word'])
            for ii in range(len(self.k_list)):
                k = self.k_list[ii]
                for iii in range(len(transaction)):
                    self.metric_list[k].eval_with_commit(
                        transaction_result[k]['commit_th_list'][iii],
                        transaction_result[k]['rank_i_c_list'][iii],
                        transaction_result[k]['rec_i_c_len_list'][iii],
                        transaction_result[k]['is_target_in_train_list'][iii],
                    )
            elapsed_time = time.time() - start_time
            for ii in range(len(pair_result)):
                item = pair_result[ii]
                target_word = item['target_word']
                if self.youngerTrace.is_younger_word(target_word):
                    self.youngerTrace.save_target_younger_predict(item)
            self.youngerTrace.trace(self.data_loader.vocab.words, transaction)
        target_younger_micro_recall = self.youngerTrace.target_younger_summary(self.k_list)
        print('younger file micro recall', target_younger_micro_recall)
        # test
        # print('target_younger_micro_recall', target_younger_micro_recall,
        #       len(self.youngerTrace.target_younger_predict_list))
        # for word in self.youngerTrace.freq:
        #     if self.youngerTrace.freq[word] > 1:
        #         print(word, self.youngerTrace.freq[word])
        # self.debug_hit_new_file_count()
        self.debug_rank_prob()
        return self.metric_list

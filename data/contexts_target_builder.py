import sys

sys.path.append('../')
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.id_mapped import IdMapped
import numpy as np
from common.util import leave_one_out
from config.config_default import get_config
from data.mode_enum import Mode
from data.sub_sampling import SubSampling
from data.negative_sampling import NegativeSampling
from data.rename_chain import RenameChain
from common.util import to_sub_word, n_gram_from_sub_word_set, get_file_level_info, sort_counter
from collections import Counter
from statistics import mean, median, stdev, variance
import time


class ContextsTargetBuilder:
    def __init__(self,
                 embeddingIndexMapped: EmbeddingIndexMapped,
                 idMapped: IdMapped,
                 subSampling: SubSampling,
                 renameChain: RenameChain,
                 mode=Mode.NORMAL,
                 is_sub_sampling=False,
                 is_subword_sub_sampling=False,
                 is_negative_sampling=False,
                 negativeSampling: NegativeSampling = None,
                 is_contexts_extend=False,
                 is_check_rename=True,
                 is_cache_context=True,
                 is_split_train_data=False,
                 is_predict_with_file_level=False,
                 remove_duplication_pre_module_data=False,
                 remove_duplication_pre_contexts=True,
                 ):
        self.embeddingIndexMapped = embeddingIndexMapped
        self.idMapped = idMapped
        self.train_id_to_word = self.idMapped.train_id_to_word
        self.mode = mode
        self.subSampling = subSampling
        self.renameChain = renameChain
        self.is_sub_sampling = is_sub_sampling
        self.is_subword_sub_sampling = is_subword_sub_sampling
        self.is_negative_sampling = is_negative_sampling
        self.negativeSampling = negativeSampling
        self.is_contexts_extend = is_contexts_extend
        self.is_check_rename = is_check_rename
        self.is_cache_context = is_cache_context
        # subword+module data = > subword and module data
        self.is_split_train_data = is_split_train_data
        self.is_predict_with_file_level = is_predict_with_file_level
        self.remove_duplication_pre_module_data = remove_duplication_pre_module_data
        self.remove_duplication_pre_contexts = remove_duplication_pre_contexts

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.negative_sampling_num = config_all['negative_sampling_num']

        # record drop
        self.drop_list = []
        self.count_list = []
        self.sub_drop_list = []
        self.sub_count_list = []

        # word => embedding_index_list
        self.context_cache = {}

    def get_train_contexts_target_split(self, train_data, train_data_commit_hash_list):
        # is_predict_with_file_level=Trueには対応していない
        # print('train_data', train_data)
        print('get_train_contexts_target_split')
        contexts_list = []
        target_list = []
        result_list = []
        max_length = 0

        for i in range(len(train_data)):
            transaction = train_data[i]
            temp_pair_list, temp_max_length = self.get_contexts_target_split_in_train(
                transaction,
                self.idMapped.train_id_to_word,
                train_data_commit_hash_list[i])
            max_length = max(max_length, temp_max_length)
            result_list.append(temp_pair_list)
        for i in range(len(result_list)):
            pair_list = result_list[i]
            for j in range(len(pair_list)):
                pair = pair_list[j]
                pair_contexts = pair['contexts']
                pair_target = pair['target']

                # padding
                padding_array = np.arange(max_length - len(pair_contexts), dtype=int)
                padding_array = np.full_like(padding_array, self.padding_id)
                # merge array
                temp_contexts = np.concatenate((pair_contexts, padding_array), axis=0)

                # print('temp_contexts', temp_contexts)
                contexts_list.append(temp_contexts)
                target_list.append(pair_target)

        return contexts_list, target_list, []

    def get_train_contexts_target(self, train_data, train_data_commit_hash_list):
        if self.mode == Mode.SUB_WORD or self.mode == Mode.N_GRAM:
            if self.is_split_train_data and \
                    not self.is_sub_sampling and \
                    not self.is_negative_sampling and \
                    not self.is_contexts_extend and \
                    not self.is_predict_with_file_level:
                return self.get_train_contexts_target_split(train_data, train_data_commit_hash_list)
        # print('train_data', train_data)
        contexts_list = []
        target_list = []
        negative_sampling_list = []
        result_list = []
        max_length = 0
        sub_max_length = 0

        for i in range(len(train_data)):
            transaction = []
            count = 0
            if not self.is_sub_sampling:
                transaction = train_data[i]
            else:
                transaction = []
                train_data_i = train_data[i]
                # sub sampling for transaction
                for j in range(len(train_data_i)):
                    item = train_data_i[j]
                    word = self.idMapped.train_id_to_word[item]
                    is_keep = self.subSampling.is_keep(word)
                    if is_keep:
                        transaction.append(item)
                    else:
                        self.drop_list.append(word)
                        count += 1
                self.count_list.append(count)
            # print('train commit hash', i, train_data_commit_hash_list[i])
            temp_pair_list, temp_max_length, temp_sub_max_length = self.get_contexts_target(
                transaction,
                self.idMapped.train_id_to_word,
                True,
                train_data_commit_hash_list[i])
            # print('temp_pair_list', temp_pair_list)
            max_length = max(max_length, temp_max_length)
            sub_max_length = max(sub_max_length, temp_sub_max_length)
            result_list.append(temp_pair_list)
        # print('result_list', result_list)
        for i in range(len(result_list)):
            pair_list = result_list[i]
            for j in range(len(pair_list)):
                pair = pair_list[j]
                pair_contexts = pair['contexts']
                pair_target = pair['target']
                pair_negative_sampling = pair['negative_sampling']
                temp_contexts = []

                if not self.is_contexts_extend:
                    # padding
                    padding_array = np.arange(max_length - len(pair_contexts), dtype=int)
                    padding_array = np.full_like(padding_array, self.padding_id)
                    # merge array
                    temp_contexts = np.concatenate((pair_contexts, padding_array), axis=0)
                else:
                    for k in range(len(pair_contexts)):
                        sub_contexts = pair_contexts[k]
                        padding_array = np.arange(sub_max_length - len(sub_contexts), dtype=int)
                        padding_array = np.full_like(padding_array, self.padding_id)
                        # merge array and append
                        temp_contexts.append(np.concatenate((sub_contexts, padding_array), axis=0))
                    padding_array = np.arange(sub_max_length, dtype=int)
                    padding_array = np.full_like(padding_array, self.padding_id)
                    for k in range(max_length - len(temp_contexts)):
                        temp_contexts.append(padding_array)
                    temp_contexts = np.array(temp_contexts)

                # negative sampling padding
                neg_padding_array = np.arange(self.negative_sampling_num - len(pair_negative_sampling), dtype=int)
                neg_padding_array = np.full_like(neg_padding_array, self.padding_id)
                # merge array
                temp_negative_sampling = np.concatenate((pair_negative_sampling, neg_padding_array), axis=0)

                # print('temp_contexts', temp_contexts)
                contexts_list.append(temp_contexts)
                target_list.append(pair_target)
                negative_sampling_list.append(temp_negative_sampling)
        # self.debug_drop()
        self.debug_sub_drop()
        return contexts_list, target_list, negative_sampling_list

    def debug_sub_drop(self):
        if self.is_subword_sub_sampling and len(self.sub_count_list) > 0:
            print('debug_sub_drop:')
            drop_counter = Counter(self.sub_drop_list)
            sort_drop_counter = sort_counter(drop_counter)
            print('drop counter len', len(drop_counter.keys()))
            # for i in range(len(sort_drop_counter)):
            #     item = sort_drop_counter[i]
            #     key = item[0]
            #     value = item[1]
            #     print(key, 'drop count %d | drop_prob %f | freq %f'
            #           % (value, self.subSampling.drop_probability(key, True),
            #              self.subSampling.sub_word_freq_list[key]))
            # for key in drop_counter.keys():
            #     print(key, drop_counter[key],
            #           round(self.subSampling.drop_probability(key, True) * 100, 2),
            #           round(self.subSampling.sub_word_freq_list[key] * 100, 2))
            print('count mean %f | median %f | stdev %f | variance %f | total %d'
                  % (mean(self.sub_count_list), median(self.sub_count_list),
                     stdev(self.sub_count_list), variance(self.sub_count_list), len(self.sub_drop_list)))

    def debug_drop(self):
        if self.is_sub_sampling and len(self.count_list) > 0:
            print('debug_drop:')
            drop_counter = Counter(self.drop_list)
            print('drop counter len', len(drop_counter.keys()))
            for key in drop_counter.keys():
                print(key, drop_counter[key],
                      round(self.subSampling.drop_probability(key) * 100, 2),
                      round(self.subSampling.word_freq_list[key] * 100, 2))
            print('count mean %f | median %f | stdev %f | variance %f | total %d'
                  % (mean(self.count_list), median(self.count_list),
                     stdev(self.count_list), variance(self.count_list), len(self.drop_list)))

    def get_contexts_target_split_in_train(self, transaction, id_to_word, commit_hash=None):
        # create contexts and target with md_id format
        # no mix subword and module data, split them to two data
        pair_list = leave_one_out(transaction)
        max_length = 0
        temp_pair_list = []

        # change format with embedding index format
        for j in range(len(pair_list)):
            pair = pair_list[j]
            target = pair['target']
            contexts = pair['contexts']
            # print('pair', target, contexts)
            temp_target = -1
            temp_contexts = []
            temp_contexts_subword = []

            # target
            target_word = id_to_word[target]
            # print('target_word', target_word)
            if target_word in self.embeddingIndexMapped.word_to_out_embedding_index:
                temp_target = self.embeddingIndexMapped.word_to_out_embedding_index[target_word]
            else:
                continue
            # contexts
            for k in range(len(contexts)):
                context = contexts[k]
                context_embedding_catch = {
                    'word': [],
                    'subword': []
                }
                # context_word is final name of context
                context_word = id_to_word[context]

                if self.is_check_rename and commit_hash and self.mode != Mode.NORMAL:
                    # check rename
                    cur_name = self.renameChain.get_cur_name_by_hash(commit_hash, context)
                else:
                    # no check rename
                    cur_name = context_word

                # get data from cache
                if self.is_cache_context is False or cur_name not in self.context_cache:
                    word_embedding_index_list, sub_word_embedding_index_list = \
                        self.embeddingIndexMapped.get_embedding_index_list_from_context_word(
                            context_word, cur_name)

                    # merge array
                    for wi in range(len(word_embedding_index_list)):
                        context_embedding_catch['word'].append(word_embedding_index_list[wi])
                    for si in range(len(sub_word_embedding_index_list)):
                        context_embedding_catch['subword'].append(sub_word_embedding_index_list[si])
                    # cache
                    self.context_cache[cur_name] = context_embedding_catch
                else:
                    # get data from cache
                    context_embedding_catch = self.context_cache[cur_name]
                for wi in range(len(context_embedding_catch['word'])):
                    temp_contexts.append(context_embedding_catch['word'][wi])
                for si in range(len(context_embedding_catch['subword'])):
                    temp_contexts_subword.append(context_embedding_catch['subword'][si])

            temp_pair_list.append({
                'target': temp_target,
                'target_embedding_index': [],
                'target_word': target_word,
                'contexts': temp_contexts,
                'negative_sampling': []
            })
            temp_pair_list.append({
                'target': temp_target,
                'target_embedding_index': [],
                'target_word': target_word,
                'contexts': temp_contexts_subword,
                'negative_sampling': []
            })
            max_length = max(max_length, len(temp_contexts))
            max_length = max(max_length, len(temp_contexts_subword))
        return temp_pair_list, max_length

    def get_contexts_target_split_in_validate(self, transaction, id_to_word, commit_hash):
        # get context embedding index if context id is old file
        # get context subword embedding index if context id is new file
        pair_list = leave_one_out(transaction)
        temp_pair_list = []

        # change format with embedding index format
        for j in range(len(pair_list)):
            pair = pair_list[j]
            target = pair['target']
            contexts = pair['contexts']

            temp_word_embedding_index_list = []
            temp_sub_word_embedding_index_list = []

            # target
            target_word = id_to_word[target]
            # print('target_word', target_word)
            if target_word in self.embeddingIndexMapped.word_to_out_embedding_index:
                temp_target = self.embeddingIndexMapped.word_to_out_embedding_index[target_word]
            else:
                temp_target = -1

            # contexts
            for k in range(len(contexts)):
                context = contexts[k]
                context_embedding_catch = []
                # context_word is final name of context
                context_word = id_to_word[context]

                if self.is_check_rename and commit_hash and self.mode != Mode.NORMAL:
                    # check rename
                    cur_name = self.renameChain.get_cur_name_by_hash(commit_hash, context)
                else:
                    # no check rename
                    cur_name = context_word

                word_embedding_index_list, sub_word_embedding_index_list = \
                    self.embeddingIndexMapped.get_embedding_index_list_from_context_word(
                        context_word, cur_name)

                for wi in range(len(word_embedding_index_list)):
                    temp_word_embedding_index_list.append(word_embedding_index_list[wi])
                if len(word_embedding_index_list) == 0:
                    for si in range(len(sub_word_embedding_index_list)):
                        temp_sub_word_embedding_index_list.append(sub_word_embedding_index_list[si])

            temp_pair_list.append({
                'target': temp_target,
                'target_embedding_index': [],
                'target_word': target_word,
                'contexts': temp_word_embedding_index_list,
                'contexts_subword': temp_sub_word_embedding_index_list,
            })
        return temp_pair_list

    def get_only_new_file(self, contexts):
        result = []
        for i in range(len(contexts)):
            item = contexts[i]
            if item not in self.train_id_to_word:
                result.append(item)
        return result

    def get_contexts_target(self,
                            transaction,
                            id_to_word,
                            is_train=True,
                            commit_hash=None,
                            is_only_new_file_context=False,
                            is_container_subword=True):
        # create contexts and target with md_id format
        # print('transaction', transaction)
        pair_list = leave_one_out(transaction)
        # print('pair_list', transaction, pair_list)
        max_length = 0
        max_sub_length = 0
        temp_pair_list = []

        count = 0

        # change format with embedding index format
        for j in range(len(pair_list)):
            pair = pair_list[j]
            target = pair['target']
            contexts = pair['contexts']
            if is_only_new_file_context:
                contexts = self.get_only_new_file(contexts)
            # print('pair', target, contexts)
            temp_target = -1
            temp_target_embedding_index_list = []
            temp_contexts = []
            temp_negative_sampling = []

            # negative sampling only for train
            # it cost long time 25s
            # is_predict_with_file_level＝Trueには対応していない
            if is_train and self.is_negative_sampling:
                sampling_list = self.negativeSampling.sampling(contexts, target)
                sampling_list_len = len(sampling_list)
                for i in range(sampling_list_len):
                    sampling_item = sampling_list[i]
                    sampling_word = id_to_word[sampling_item]
                    if sampling_word in self.embeddingIndexMapped.word_to_out_embedding_index:
                        temp_negative_sampling.append(
                            self.embeddingIndexMapped.word_to_out_embedding_index[sampling_word])
                # print('temp_negative_sampling', temp_negative_sampling)
            # target
            if self.is_predict_with_file_level:
                target_word = get_file_level_info(id_to_word[target])
            else:
                target_word = id_to_word[target]
            # print('target_word', target_word)
            if target_word in self.embeddingIndexMapped.word_to_out_embedding_index:
                temp_target = self.embeddingIndexMapped.word_to_out_embedding_index[target_word]
            else:
                # can't build target embedding index
                # print('skip', target_word)
                if is_train:
                    # in training step, skip
                    continue
                else:
                    # in validation step, just set target as -1
                    temp_target = -1
            # print('temp_target', temp_target, target_word)
            # is_predict_with_file_level＝Trueには対応していない
            if not self.is_predict_with_file_level:
                target_word_embedding_index_list, target_sub_word_embedding_index_list = \
                    self.embeddingIndexMapped.get_embedding_index_list_from_target_word(
                        target_word)
                # only using sub word information, this target embedding index info will use in predict new file
                temp_target_embedding_index_list = target_sub_word_embedding_index_list
            else:
                temp_target_embedding_index_list = [temp_target]

            # contexts
            for k in range(len(contexts)):
                context = contexts[k]
                context_embedding_catch = []
                # context_word is final name of context
                context_word = id_to_word[context]

                if self.is_check_rename and commit_hash and self.mode != Mode.NORMAL:
                    # check rename
                    cur_name = self.renameChain.get_cur_name_by_hash(commit_hash, context)
                    # test
                    # final_name = id_to_word[context]
                    # if final_name != cur_name:
                    #     print(commit_hash, k, context, contexts, transaction, final_name, cur_name)
                else:
                    # no check rename
                    cur_name = context_word

                # not need to get from cache
                if is_container_subword is False:
                    context_embedding_catch = []
                    word_embedding_index = \
                        self.embeddingIndexMapped.get_embedding_index_from_context_word(context_word)
                    if word_embedding_index != -1:
                        context_embedding_catch.append(word_embedding_index)
                else:
                    # get data from cache
                    if self.is_cache_context is False or cur_name not in self.context_cache:
                        word_embedding_index_list, sub_word_embedding_index_list = \
                            self.embeddingIndexMapped.get_embedding_index_list_from_context_word(
                                context_word, cur_name)
                        if self.remove_duplication_pre_module_data:
                            sub_word_embedding_index_list = list(set(sub_word_embedding_index_list))
                        result_sub_word_embedding_index_list = []

                        current_contexts_embedding_index_list = []

                        if self.is_subword_sub_sampling and len(sub_word_embedding_index_list) > 0:
                            # shrink sub_word_embedding_index_list
                            for i in range(len(sub_word_embedding_index_list)):
                                sub_word_index = sub_word_embedding_index_list[i]
                                sub_word = self.embeddingIndexMapped.in_embedding_index_to_word[sub_word_index]
                                is_keep = self.subSampling.is_keep(sub_word, True)
                                if is_keep:
                                    result_sub_word_embedding_index_list.append(sub_word_index)
                                else:
                                    count += 1
                                    self.sub_drop_list.append(sub_word)
                        else:
                            result_sub_word_embedding_index_list = sub_word_embedding_index_list

                        if self.is_contexts_extend:
                            # merge array
                            for wi in range(len(word_embedding_index_list)):
                                current_contexts_embedding_index_list.append(word_embedding_index_list[wi])
                            for si in range(len(result_sub_word_embedding_index_list)):
                                current_contexts_embedding_index_list.append(result_sub_word_embedding_index_list[si])
                            max_sub_length = max(max_sub_length, len(current_contexts_embedding_index_list))
                            context_embedding_catch = current_contexts_embedding_index_list
                        else:
                            # merge array
                            for wi in range(len(word_embedding_index_list)):
                                context_embedding_catch.append(word_embedding_index_list[wi])
                            for si in range(len(result_sub_word_embedding_index_list)):
                                context_embedding_catch.append(result_sub_word_embedding_index_list[si])
                        # cache
                        self.context_cache[cur_name] = context_embedding_catch
                    else:
                        # get data from cache
                        context_embedding_catch = self.context_cache[cur_name]

                if self.is_contexts_extend:
                    temp_contexts.append(context_embedding_catch)
                else:
                    # if not is_train:
                    #     print('context_embedding_catch', context_embedding_catch)
                    for ci in range(len(context_embedding_catch)):
                        temp_contexts.append(context_embedding_catch[ci])

            # print('temp_contexts', temp_contexts)
            # print()
            # delete repeating index, but it drop recall
            # temp_contexts = list(set(temp_contexts))
            if self.remove_duplication_pre_contexts:
                temp_contexts = list(set(temp_contexts))
            temp_pair_list.append({
                'target': temp_target,
                'target_embedding_index': temp_target_embedding_index_list,
                'target_word': target_word,
                'contexts': temp_contexts,
                'negative_sampling': temp_negative_sampling
            })
            max_length = max(max_length, len(temp_contexts))
            self.sub_count_list.append(count)
        return temp_pair_list, max_length, max_sub_length

    # don't use this function
    # def get_validate_contexts_target(self, validate_data):
    #     # print('validate_data', validate_data)
    #     max_sub_length = 0
    #     temp_result_list = []
    #     result_list = []
    #     # print('validate_data', validate_data)
    #     for i in range(len(validate_data)):
    #         transaction = validate_data[i]
    #         # print('transaction', transaction)
    #         temp_pair_list, temp_max_length, temp_max_sub_length = \
    #             self.get_contexts_target(transaction, self.idMapped.all_id_to_word, False, i)
    #         temp_result_list.append(temp_pair_list)
    #         max_sub_length = max(max_sub_length, temp_max_sub_length)
    #     for i in range(len(temp_result_list)):
    #         pair_list = temp_result_list[i]
    #         for j in range(len(pair_list)):
    #             pair = pair_list[j]
    #             pair_contexts = pair['contexts']
    #             pair_target = pair['target']
    #             temp_contexts = []
    #
    #             if not self.is_contexts_extend:
    #                 temp_contexts = pair_contexts
    #             else:
    #                 for k in range(len(pair_contexts)):
    #                     sub_contexts = pair_contexts[k]
    #                     print('sub_contexts len', len(sub_contexts))
    #                     padding_array = np.arange(max_sub_length - len(sub_contexts), dtype=int)
    #                     padding_array = np.full_like(padding_array, self.padding_id)
    #                     # merge array and append
    #                     temp_contexts.append(np.concatenate((sub_contexts, padding_array), axis=0))
    #
    #             # print('temp_contexts', temp_contexts)
    #             result_list.append({
    #                 'contexts': temp_contexts,
    #                 'target': pair_target
    #             })
    #     return result_list

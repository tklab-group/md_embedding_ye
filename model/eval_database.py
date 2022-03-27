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
from data.mongo import PredictDao
from data.mongo import YoungerResultDao, YoungerContextsResultDao, LowFreqResultDao, LowFreqContextsResultDao
from data.data_store import DataStore
from data.data_loader import DataLoader
from data.younger_trace import YoungerTrace
from data.low_freq_trace import LowFreqTrace
from common.util import leave_one_out, save_boxplot
from data.delete_record import DeleteRecord
from data.git_name_version import get_git_name_version
from data.tarmaq_result import get_tarmaq_result
from common.util import save_data, load_data
import os
from data.util import load_predict_result


def save_result(data, file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    save_data(data, pkl_path)


def load_result(file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    result = load_data(pkl_path)
    return result


def rank(word_list, target_word):
    for i in range(len(word_list)):
        if word_list[i] == target_word:
            return i + 1
    return 0


def get_top_k(word_list, k):
    if len(word_list) < k:
        return word_list
    return word_list[0:k]


def print_recall(k, micro_recall, macro_recall):
    print('k %d | micro recall %.3f | macro recall %.3f'
          % (k, micro_recall, macro_recall))


class EvaluationDataBase:
    def __init__(self,
                 git_name,
                 version,
                 sub_version,
                 expected_validate_length,
                 mode_str
                 ):
        print('EvaluationDataBase param',
              git_name,
              version,
              sub_version,
              expected_validate_length,
              mode_str)
        self.git_name = git_name
        self.version = version
        self.sub_version = sub_version
        self.expected_validate_length = expected_validate_length
        self.mode_str = mode_str
        self.dao = PredictDao()
        self.k_list = [1, 5, 10, 15, 20]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()
        self.predict_list = []
        self.load_data()
        self.dataStore = DataStore()
        # self.youngerResultDao = YoungerResultDao()
        # self.youngerContextsResultDao = YoungerContextsResultDao()
        # self.lowFreqResultDao = LowFreqResultDao()
        # self.lowFreqContextsResultDao = LowFreqContextsResultDao()

    def count_no_recommend(self):
        query_list_temp = load_predict_result(self.git_name, self.version)
        total = 0
        no_recommend = 0
        no_contexts = 0
        print(len(query_list_temp))
        for i in range(len(query_list_temp)):
            doc = query_list_temp[i]
            predict_result = doc['predict_result']
            commit_th = doc['commit_th']
            commit_hash = doc['commit_hash']
            for i2 in range(len(predict_result)):
                # total += 1
                item = predict_result[i2]
                # target contexts is embedding index format
                target = item['target']
                target_word = item['target_word']
                contexts = item['contexts']
                top100_aq = item['top100_aq']
                top100_prob_list = item['top100_prob_list']
                if target != -1:
                    total += 1
                    if len(contexts) == 0:
                        no_contexts += 1
                    if len(top100_aq) == 0:
                        no_recommend += 1
        print(self.git_name, total, no_contexts, no_recommend)

    def validate_low_freq(self, mode, threshold, most_recent=5000):
        data_loader = DataLoader(
            git_name=self.git_name,
            expected_validate_length=self.expected_validate_length,
            most_recent=most_recent,
            dataStore=self.dataStore,
            mode=mode,
            deleteRecord=None
        )
        validate_data = data_loader.validate_data
        lowFreqTrace = LowFreqTrace(self.git_name, self.expected_validate_length, most_recent, threshold)
        all_id_to_word = lowFreqTrace.all_id_to_word
        # query_list = self.dao.query_by(self.git_name, self.version, self.mode_str)
        # query_list_temp = []
        # for doc in query_list:
        #     query_list_temp.append(doc)
        query_list_temp = load_predict_result(self.git_name, self.version)
        print(len(query_list_temp), self.expected_validate_length,
              'is same as', len(query_list_temp) == self.expected_validate_length)
        start_time = time.time()
        for i in range(len(query_list_temp)):
            doc = query_list_temp[i]
            predict_result = doc['predict_result']
            commit_th = doc['commit_th']
            commit_hash = doc['commit_hash']
            # transaction_md is md id format
            transaction_md = validate_data[i]
            pair_list_md = leave_one_out(transaction_md)
            lowFreqTrace.trace(self.expected_validate_length - i)
            for i2 in range(len(predict_result)):
                item = predict_result[i2]
                # target contexts is embedding index format
                target = item['target']
                target_word = item['target_word']
                contexts = item['contexts']
                # target_md, contexts_md is md id format
                target_md = pair_list_md[i2]['target']
                contexts_md = pair_list_md[i2]['contexts']
                target_md_word = all_id_to_word[target_md]

                if target_word != target_md_word:
                    print('error', target_word, target_md_word)
                top100_aq = item['top100_aq']
                top100_prob_list = item['top100_prob_list']
                contexts_component_result = lowFreqTrace.contexts_component(contexts_md)
                lowFreqTrace.save_contexts_component_predict({
                    'contexts_component': contexts_component_result,
                    'predict_result': item
                })
                # print('contexts_component_result', contexts_md, contexts_component_result['new_list'])
                self.lowFreqContextsResultDao.insert({
                    'predict_result': item,
                    'commit_th': commit_th,
                    'commit_hash': commit_hash,
                    'git_name': self.git_name,
                    'cur_expected_validate_length': self.expected_validate_length - i,
                    'most_recent': most_recent,
                    'mode': self.mode_str,
                    'version': self.version + self.sub_version,
                    'new_rate': str(contexts_component_result['new_rate']),
                    'new_list': contexts_component_result['new_list'],
                    'old_rate': str(contexts_component_result['old_rate']),
                    'old_list': contexts_component_result['old_list'],
                    'low_freq_rate': str(contexts_component_result['low_freq_rate']),
                    'low_freq_list': contexts_component_result['low_freq_list'],
                    'threshold': threshold
                })
                if lowFreqTrace.is_low_freq_word(target_word):
                    lowFreqTrace.save_target_low_freq_predict(item)
                    self.lowFreqResultDao.insert({
                        'predict_result': item,
                        'commit_th': commit_th,
                        'commit_hash': commit_hash,
                        'git_name': self.git_name,
                        'cur_expected_validate_length': self.expected_validate_length - i,
                        'most_recent': most_recent,
                        'mode': self.mode_str,
                        'version': self.version + self.sub_version,
                        'threshold': threshold
                    })

            if i % 100 == 0:
                print('end with:', i, 'cost time:', time.time() - start_time)
        micro_recall = lowFreqTrace.target_low_freq_summary(self.k_list)
        # save_result(lowFreqTrace.target_low_freq_predict_list,
        #           self.git_name + '_' + self.version + self.sub_version + '_inactive_target_' + str(threshold))
        # save_result(lowFreqTrace.contexts_component_predict_list,
        #           self.git_name + '_' + self.version + self.sub_version + '_contexts_component_' + str(threshold))
        print('low freq file micro recall', micro_recall)

    def validate_younger(self, mode, most_recent=5000):
        data_loader = DataLoader(
            git_name=self.git_name,
            expected_validate_length=self.expected_validate_length,
            most_recent=most_recent,
            dataStore=self.dataStore,
            mode=mode
        )
        validate_data = data_loader.validate_data
        all_id_to_word = data_loader.idMapped.all_id_to_word
        youngerTrace = YoungerTrace(all_id_to_word)
        query_list = self.dao.query_by(self.git_name, self.version, self.mode_str)
        query_list_temp = []
        for doc in query_list:
            query_list_temp.append(doc)
        print(len(query_list_temp), self.expected_validate_length,
              'is same as', len(query_list_temp) == self.expected_validate_length)
        start_time = time.time()
        for i in range(len(query_list_temp)):
            doc = query_list_temp[i]
            predict_result = doc['predict_result']
            commit_th = doc['commit_th']
            commit_hash = doc['commit_hash']
            # transaction_md is md id format
            transaction_md = validate_data[i]
            pair_list_md = leave_one_out(transaction_md)
            for i2 in range(len(predict_result)):
                item = predict_result[i2]
                # target contexts is embedding index format
                target = item['target']
                target_word = item['target_word']
                contexts = item['contexts']
                # target_md, contexts_md is md id format
                target_md = pair_list_md[i2]['target']
                contexts_md = pair_list_md[i2]['contexts']
                target_md_word = all_id_to_word[target_md]

                if target_word != target_md_word:
                    print('error', target_word, target_md_word)
                top100_aq = item['top100_aq']
                top100_prob_list = item['top100_prob_list']
                contexts_component_result = youngerTrace.contexts_component(contexts_md)
                youngerTrace.save_contexts_component_predict({
                    'contexts_component': contexts_component_result,
                    'predict_result': item
                })
                # print('contexts_component_result', contexts_md, contexts_component_result['new_list'])
                self.youngerContextsResultDao.insert({
                    'predict_result': item,
                    'commit_th': commit_th,
                    'commit_hash': commit_hash,
                    'git_name': self.git_name,
                    'cur_expected_validate_length': self.expected_validate_length - i,
                    'most_recent': most_recent,
                    'mode': self.mode_str,
                    # 'vocab_list': list(youngerTrace.vocab_set),
                    'version': self.version + self.sub_version,
                    'vocab_list_len': len(youngerTrace.vocab_set),
                    # 'contexts_component': contexts_component_result,
                    'new_rate': str(contexts_component_result['new_rate']),
                    'new_list': contexts_component_result['new_list'],
                    'old_rate': str(contexts_component_result['old_rate']),
                    'old_list': contexts_component_result['old_list'],
                    'younger_rate': str(contexts_component_result['younger_rate']),
                    'younger_list': contexts_component_result['younger_list'],
                })
                if youngerTrace.is_younger_word(target_word):
                    youngerTrace.save_target_younger_predict(item)
                    cur_freq = []
                    for key in youngerTrace.freq:
                        cur_freq.append({
                            'key': key,
                            'value': youngerTrace.freq[key]
                        })
                    self.youngerResultDao.insert({
                        'predict_result': item,
                        'commit_th': commit_th,
                        'commit_hash': commit_hash,
                        'git_name': self.git_name,
                        'cur_expected_validate_length': self.expected_validate_length - i,
                        'most_recent': most_recent,
                        'mode': self.mode_str,
                        'freq': cur_freq,
                        # 'vocab_list': list(youngerTrace.vocab_set),
                        'version': self.version + self.sub_version,
                        'vocab_list_len': len(youngerTrace.vocab_set)
                    })
            cur_data_loader = DataLoader(
                git_name=self.git_name,
                expected_validate_length=self.expected_validate_length - i,
                most_recent=most_recent,
                dataStore=self.dataStore,
                mode=mode
            )
            transaction = validate_data[i]
            youngerTrace.trace(cur_data_loader.vocab.words, transaction)
            if i % 100 == 0:
                print('end with:', i, 'cost time:', time.time() - start_time)
        target_younger_micro_recall = youngerTrace.target_younger_summary(self.k_list)
        print('younger file micro recall', target_younger_micro_recall)

    def load_data(self):
        query_list = self.dao.query_by(self.git_name, self.version, self.mode_str)
        for doc in query_list:
            predict_result = doc['predict_result']
            commit_th = doc['commit_th']
            commit_hash = doc['commit_hash']
            for i in range(len(predict_result)):
                item = predict_result[i]
                target = item['target']
                target_embedding_index = item['target_embedding_index']
                target_word = item['target_word']
                contexts = item['contexts']
                # if len(contexts) == 0:
                #     print('bingo')
                # negative_sampling = item['negative_sampling']
                top100_aq = item['top100_aq']
                top100_prob_list = item['top100_prob_list']
                if target != -1:
                    is_target_in_train = True
                else:
                    is_target_in_train = False
                for ii in range(len(self.k_list)):
                    k = self.k_list[ii]
                    top_k = get_top_k(top100_aq, k)
                    rank_i_c = rank(top_k, target)
                    rec_i_c_len = len(top_k)
                    self.metric_list[k].eval_with_commit(
                        commit_th,
                        rank_i_c,
                        rec_i_c_len,
                        is_target_in_train
                    )

    def validate(self):
        result = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            result[k] = {
                'micro': 0,
                'macro': 0
            }
        for i in range(len(self.metric_list)):
            k = self.k_list[i]

            metric = self.metric_list[k]
            # is_consider_new_file=False
            micro_recall, macro_recall = metric.summary(True)
            result[k]['micro'] = micro_recall
            result[k]['macro'] = macro_recall
            print_recall(k, micro_recall, macro_recall)
        return result

    def check_result_list(self, result_list):
        micro_recall_k = {}
        macro_recall_k = {}
        micro_mean_k = {}
        macro_mean_k = {}
        micro_stdev_k = {}
        macro_stdev_k = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            micro_recall_k[k] = []
            macro_recall_k[k] = []
        for i in range(len(result_list)):
            result = result_list[i]
            # print('result', result)
            for j in range(len(self.k_list)):
                k = self.k_list[j]
                micro_recall = result[k]['micro']
                macro_recall = result[k]['macro']
                micro_recall_k[k].append(micro_recall)
                macro_recall_k[k].append(macro_recall)
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            micro_mean_k[k] = mean(micro_recall_k[k])
            macro_mean_k[k] = mean(macro_recall_k[k])
            micro_stdev_k[k] = stdev(micro_recall_k[k])
            macro_stdev_k[k] = stdev(macro_recall_k[k])
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            print(
                k,
                'micro_mean',
                round(micro_mean_k[k], 3),
                'macro_mean',
                round(macro_mean_k[k], 3),
                'micro_stdev',
                round(micro_stdev_k[k], 3),
                'macro_stdev',
                round(macro_stdev_k[k], 3))

    def save_fig(self, git_name, mode_str, result_list, tarmaq_result):
        micro_recall_k = {}
        macro_recall_k = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            micro_recall_k[k] = []
            macro_recall_k[k] = []
        for i in range(len(result_list)):
            result = result_list[i]
            # print('result', result)
            for j in range(len(self.k_list)):
                k = self.k_list[j]
                micro_recall = result[k]['micro']
                macro_recall = result[k]['macro']
                micro_recall_k[k].append(micro_recall)
                macro_recall_k[k].append(macro_recall)
        print('tarmaq_result', tarmaq_result)
        model_name = 'CBOW model'
        if mode_str == 'subword':
            model_name = 'Subword model'
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            tarmaq_micro_recall_k = tarmaq_result[str(k)]['micro']
            tarmaq_macro_recall_k = tarmaq_result[str(k)]['macro']
            save_boxplot(git_name,
                         model_name,
                         'micro recall@' + str(k),
                         mode_str + '_micro_recall@' + str(k),
                         tarmaq_micro_recall_k,
                         micro_recall_k[k])
            save_boxplot(git_name,
                         model_name,
                         'macro recall@' + str(k),
                         mode_str + '_macro_recall@' + str(k),
                         tarmaq_macro_recall_k,
                         macro_recall_k[k])

    def print_tex_table(self, result_list, tarmaq_result):
        micro_recall_k = {}
        macro_recall_k = {}
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            micro_recall_k[k] = []
            macro_recall_k[k] = []
        for i in range(len(result_list)):
            result = result_list[i]
            # print('result', result)
            for j in range(len(self.k_list)):
                k = self.k_list[j]
                micro_recall = result[k]['micro']
                macro_recall = result[k]['macro']
                micro_recall_k[k].append(micro_recall)
                macro_recall_k[k].append(macro_recall)
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            tarmaq_micro_recall_k = tarmaq_result[str(k)]['micro']
            tarmaq_macro_recall_k = tarmaq_result[str(k)]['macro']
            # 1    & 0 & 0 & 0.211 & 0.176 \\ \hline
            print(
                k,
                '&',
                round(tarmaq_micro_recall_k, 3),
                '&',
                round(mean(micro_recall_k[k]), 3),
                '&',
                round(tarmaq_macro_recall_k, 3),
                '&',
                round(mean(macro_recall_k[k]), 3),
                '\\\\',
                '\hline'
            )


def print_single_tex_table(tarmaq_result, normal_result, subword_result, k_list):
    print()
    print()
    for i in range(len(k_list)):
        k = k_list[i]
        tarmaq_micro_recall_k = round(tarmaq_result[str(k)]['micro'], 3)
        tarmaq_macro_recall_k = round(tarmaq_result[str(k)]['macro'], 3)
        normal_micro_recall_k = round(normal_result[k]['micro'], 3)
        subword_micro_recall_k = round(subword_result[k]['micro'], 3)
        normal_macro_recall_k = round(normal_result[k]['macro'], 3)
        subword_macro_recall_k = round(subword_result[k]['macro'], 3)
        print(
            k,
            '&',
            bf_value(tarmaq_micro_recall_k, normal_micro_recall_k, subword_micro_recall_k),
            '&',
            bf_value(normal_micro_recall_k, tarmaq_micro_recall_k, subword_micro_recall_k),
            '&',
            bf_value(subword_micro_recall_k, tarmaq_micro_recall_k, normal_micro_recall_k),
            '&',
            bf_value(tarmaq_macro_recall_k, normal_macro_recall_k, subword_macro_recall_k),
            '&',
            bf_value(normal_macro_recall_k, tarmaq_macro_recall_k, subword_macro_recall_k),
            '&',
            bf_value(subword_macro_recall_k, tarmaq_macro_recall_k, normal_macro_recall_k),
            '\\\\',
            '\hline'
        )


def bf_value(value1, value2, value3):
    if value1 >= value2 and value1 >= value3:
        return '\\bf{' + str(value1) + '}'
    else:
        return str(value1)


def print_total_tex_table(tarmaq_result, normal_result_list, subword_result_list, k_list):
    normal_micro_recall_k = {}
    normal_macro_recall_k = {}
    subword_micro_recall_k = {}
    subword_macro_recall_k = {}
    for i in range(len(k_list)):
        k = k_list[i]
        normal_micro_recall_k[k] = []
        normal_macro_recall_k[k] = []
        subword_micro_recall_k[k] = []
        subword_macro_recall_k[k] = []
    for i in range(len(normal_result_list)):
        normal_result = normal_result_list[i]
        for j in range(len(k_list)):
            k = k_list[j]
            normal_micro_recall_k[k].append(normal_result[k]['micro'])
            normal_macro_recall_k[k].append(normal_result[k]['macro'])
    for i in range(len(subword_result_list)):
        subword_result = subword_result_list[i]
        # print('result', result)
        for j in range(len(k_list)):
            k = k_list[j]
            subword_micro_recall_k[k].append(subword_result[k]['micro'])
            subword_macro_recall_k[k].append(subword_result[k]['macro'])
    for i in range(len(k_list)):
        k = k_list[i]
        tarmaq_micro_recall_k = round(tarmaq_result[str(k)]['micro'], 3)
        tarmaq_macro_recall_k = round(tarmaq_result[str(k)]['macro'], 3)
        # 1    & 0 & 0 & 0.211 & 0.176 \\ \hline
        mean_normal_micro_recall_k = round(mean(normal_micro_recall_k[k]), 3)
        mean_subword_micro_recall_k = round(mean(subword_micro_recall_k[k]), 3)
        mean_normal_macro_recall_k = round(mean(normal_macro_recall_k[k]), 3)
        mean_subword_macro_recall_k = round(mean(subword_macro_recall_k[k]), 3)
        # max value become \bf{value}
        print(
            k,
            '&',
            bf_value(tarmaq_micro_recall_k, mean_normal_micro_recall_k, mean_subword_micro_recall_k),
            '&',
            bf_value(mean_normal_micro_recall_k, tarmaq_micro_recall_k, mean_subword_micro_recall_k),
            '&',
            bf_value(mean_subword_micro_recall_k, tarmaq_micro_recall_k, mean_normal_micro_recall_k),
            '&',
            bf_value(tarmaq_macro_recall_k, mean_normal_macro_recall_k, mean_subword_macro_recall_k),
            '&',
            bf_value(mean_normal_macro_recall_k, tarmaq_macro_recall_k, mean_subword_macro_recall_k),
            '&',
            bf_value(mean_subword_macro_recall_k, tarmaq_macro_recall_k, mean_normal_macro_recall_k),
            '\\\\',
            '\hline'
        )


def get_result_list_fix_seed(git_name, mode_str):
    dummy = {
        1: {'micro': 0, 'macro': 0},
        5: {'micro': 0, 'macro': 0},
        10: {'micro': 0, 'macro': 0},
        15: {'micro': 0, 'macro': 0},
        20: {'micro': 0, 'macro': 0},
    }
    version_list = get_git_name_version(git_name, mode_str)
    if len(version_list) == 0:
        return dummy
    version = version_list[0]['version']
    sub_version = version_list[0]['sub_version']
    mode = version_list[0]['mode']
    str_mode = 'NORMAL'
    if mode == Mode.NORMAL:
        str_mode = 'NORMAL'
    elif mode == Mode.SUB_WORD:
        str_mode = 'SUB_WORD'
    elif mode == Mode.SUB_WORD_NO_FULL:
        str_mode = 'SUB_WORD_NO_FULL'
    else:
        str_mode = 'N_GRAM'
    eval = EvaluationDataBase(git_name, version, sub_version, 1000, str_mode)
    result = eval.validate()
    return result


def get_result_list(git_name, mode_str):
    version_list = get_git_name_version(git_name, mode_str)
    tarmaq_result = get_tarmaq_result(git_name)
    result_list = []
    for i in range(len(version_list)):
        version = version_list[i]['version']
        sub_version = version_list[i]['sub_version']
        mode = version_list[i]['mode']
        str_mode = 'NORMAL'
        if mode == Mode.NORMAL:
            str_mode = 'NORMAL'
        elif mode == Mode.SUB_WORD:
            str_mode = 'SUB_WORD'
        elif mode == Mode.SUB_WORD_NO_FULL:
            str_mode = 'SUB_WORD_NO_FULL'
        else:
            str_mode = 'N_GRAM'
        print(version, sub_version, str_mode)
        eval = EvaluationDataBase(git_name, version, sub_version, 1000, str_mode)
        result = eval.validate()
        result_list.append(result)
        # eval.validate_younger(mode=Mode.NORMAL, most_recent=5000)
        # eval.validate_low_freq(mode=Mode.NORMAL, most_recent=5000)
        print()
    print('result_list', result_list)
    eval.check_result_list(result_list)
    mode_str = ''
    mode = version_list[0]['mode']
    if mode == Mode.NORMAL:
        mode_str = 'normal'
    elif mode == Mode.SUB_WORD:
        mode_str = 'subword'
    elif mode == Mode.SUB_WORD_NO_FULL:
        mode_str = 'subword_no_full'
    else:
        mode_str = 'n_gram'
    eval.save_fig(git_name, mode_str, result_list, tarmaq_result)
    eval.print_tex_table(result_list, tarmaq_result)
    return result_list


def save_low_freq_data(git_name, mode_str, threshold):
    print('start', git_name, mode_str)
    version_list = get_git_name_version(git_name, mode_str)
    version = version_list[0]['version']
    # sub_version = version_list[0]['sub_version']
    sub_version = '_1_20_paper_' + str(threshold)
    mode = version_list[0]['mode']
    str_mode = 'NORMAL'
    if mode == Mode.NORMAL:
        str_mode = 'NORMAL'
    elif mode == Mode.SUB_WORD:
        str_mode = 'SUB_WORD'
    elif mode == Mode.SUB_WORD_NO_FULL:
        str_mode = 'SUB_WORD_NO_FULL'
    else:
        str_mode = 'N_GRAM'
    print(version, sub_version, str_mode)
    eval = EvaluationDataBase(git_name, version, sub_version, 1000, str_mode)
    eval.validate_low_freq(mode=Mode.NORMAL, most_recent=5000, threshold=threshold)
    print('end', git_name, mode)
    print()
    print()


def low_freq_main():
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    mode_str_list = ['normal', 'subword']
    # first 3
    threshold_list = [3]
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        for i2 in range(len(mode_str_list)):
            mode_str = mode_str_list[i2]
            for i3 in range(len(threshold_list)):
                threshold = threshold_list[i3]
                save_low_freq_data(git_name, mode_str, threshold)


def main():
    git_name = 'cassandra'
    tarmaq_result = get_tarmaq_result(git_name)
    normal_result_list = get_result_list(git_name, 'normal')
    subword_result_list = get_result_list(git_name, 'subword')
    dummy = {
        1: {'micro': 0, 'macro': 0},
        5: {'micro': 0, 'macro': 0},
        10: {'micro': 0, 'macro': 0},
        15: {'micro': 0, 'macro': 0},
        20: {'micro': 0, 'macro': 0},
    }
    # subword_result_list = [dummy]
    k_list = [1, 5, 10, 15, 20]
    print_total_tex_table(tarmaq_result, normal_result_list, subword_result_list, k_list)


def main_single():
    git_name = 'tomcat'
    tarmaq_result = get_tarmaq_result(git_name)
    normal_result_list = get_result_list_fix_seed(git_name, 'normal')
    subword_result_list = get_result_list_fix_seed(git_name, 'subword')
    k_list = [1, 5, 10, 15, 20]
    print(normal_result_list, subword_result_list)
    print_single_tex_table(tarmaq_result, normal_result_list, subword_result_list, k_list)


def count_no_recommend_main():
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    mode_str_list = ['normal', 'subword']
    # first 3
    threshold_list = [3]
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        for i2 in range(len(mode_str_list)):
            mode_str = mode_str_list[i2]
            print('start', git_name, mode_str)
            version_list = get_git_name_version(git_name, mode_str)
            version = version_list[0]['version']
            sub_version = '_1_20_paper_' + str(3)
            mode = version_list[0]['mode']
            str_mode = 'NORMAL'
            if mode == Mode.NORMAL:
                str_mode = 'NORMAL'
            elif mode == Mode.SUB_WORD:
                str_mode = 'SUB_WORD'
            elif mode == Mode.SUB_WORD_NO_FULL:
                str_mode = 'SUB_WORD_NO_FULL'
            else:
                str_mode = 'N_GRAM'
            print(version, sub_version, str_mode)
            eval = EvaluationDataBase(git_name, version, sub_version, 1000, str_mode)
            eval.count_no_recommend()
            print('end', git_name, mode)
            print()
            print()


if __name__ == '__main__':
    # main_single()
    main()
    # low_freq_main()
    # count_no_recommend_main()
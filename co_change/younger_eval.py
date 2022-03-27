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
from data.mongo import YoungerTarmaqResultDao, YoungerTarmaqContextsResultDao
from data.data_store import DataStore
from data.data_loader import DataLoader
from data.younger_trace import YoungerTrace
from data.util import get_co_change
from co_change.handle_target import HandleTarget
from common.util import leave_one_out


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
    print('k %d | micro recall %.2f | macro recall %.2f'
          % (k, micro_recall, macro_recall))


class YoungerEval:
    def __init__(self,
                 git_name,
                 version,
                 sub_version,
                 expected_validate_length,
                 most_recent
                 ):
        self.git_name = git_name
        self.version = version
        self.sub_version = sub_version
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent
        self.dao = PredictDao()
        self.k_list = [1, 5, 10, 15, 20]
        self.predict_list = []
        self.youngerTarmaqResultDao = YoungerTarmaqResultDao()
        self.youngerTarmaqContextsResultDao = YoungerTarmaqContextsResultDao()
        self.co_change = None
        self.handleTarget = None
        self.dataStore = DataStore()
        self.load_data()

    def load_data(self):
        self.handleTarget = HandleTarget(self.git_name, self.expected_validate_length, self.most_recent)
        git_name_false = self.git_name + '_false'
        if self.most_recent > 0:
            git_name_false += '_' + str(self.most_recent)
        self.co_change = get_co_change(git_name_false)
        self.co_change = self.handleTarget.filter(self.co_change, False)

    def validate_younger(self):
        data_loader = DataLoader(
            git_name=self.git_name,
            expected_validate_length=self.expected_validate_length,
            most_recent=self.most_recent,
            dataStore=self.dataStore,
        )
        validate_data = data_loader.validate_data
        all_id_to_word = data_loader.idMapped.all_id_to_word
        youngerTrace = YoungerTrace(all_id_to_word)

        print(len(self.co_change), self.expected_validate_length,
              'is same as', len(self.co_change) == self.expected_validate_length)
        start_time = time.time()

        for i in range(len(self.co_change)):
            transaction_md = validate_data[i]
            pair_list_md = leave_one_out(transaction_md)

            pair_list = self.co_change[i]['list']
            commit_th = i
            for i2 in range(len(pair_list)):
                pair = pair_list[i2]
                # print('pair', pair)
                contexts = pair['contexts']
                target = pair['target']
                topKList = pair['topKList']

                # target_md, contexts_md is md id format
                target_md = pair_list_md[i2]['target']
                contexts_md = pair_list_md[i2]['contexts']

                # if target != target_md:
                #     print('error', target, target_md)
                if len(contexts) != len(contexts_md):
                    print('contexts error', contexts, contexts_md)

                contexts_component_result = youngerTrace.contexts_component(contexts_md)
                youngerTrace.save_contexts_component_predict({
                    'contexts_component': contexts_component_result,
                    'predict_result': pair
                })
                self.youngerTarmaqContextsResultDao.insert({
                    'predict_result': pair,
                    'commit_th': commit_th,
                    'git_name': self.git_name,
                    'cur_expected_validate_length': self.expected_validate_length - i,
                    'most_recent': self.most_recent,
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
                if youngerTrace.is_younger_md_id(target):
                    youngerTrace.save_target_younger_predict({
                        'target': target,
                        'top100_aq': topKList
                    })
                    cur_freq = []
                    for key in youngerTrace.freq:
                        cur_freq.append({
                            'key': key,
                            'value': youngerTrace.freq[key]
                        })
                    self.youngerTarmaqResultDao.insert({
                        'predict_result': pair,
                        'commit_th': commit_th,
                        'git_name': self.git_name,
                        'cur_expected_validate_length': self.expected_validate_length - i,
                        'most_recent': self.most_recent,
                        'freq': cur_freq,
                        # 'vocab_list': list(youngerTrace.vocab_set),
                        'version': self.version + self.sub_version,
                        'vocab_list_len': len(youngerTrace.vocab_set)
                    })
            cur_data_loader = DataLoader(
                git_name=self.git_name,
                expected_validate_length=self.expected_validate_length - i,
                most_recent=self.most_recent,
                dataStore=self.dataStore
            )
            transaction = validate_data[i]
            youngerTrace.trace(cur_data_loader.vocab.words, transaction)
            if i % 100 == 0:
                print('end with:', i, 'cost time:', time.time() - start_time)
        target_younger_micro_recall = youngerTrace.target_younger_summary(self.k_list)
        print('tarmaq younger file micro recall', target_younger_micro_recall)


if __name__ == '__main__':
    print('test')
    eval = YoungerEval('tomcat', 'tomcat_v4', '_younger_4', 1000, 5000)
    eval.validate_younger()


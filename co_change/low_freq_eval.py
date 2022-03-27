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
from data.mongo import LowFreqTarmaqResultDao, LowFreqTarmaqContextsResultDao
from data.data_store import DataStore
from data.data_loader import DataLoader
from data.low_freq_trace import LowFreqTrace
from data.util import get_co_change
from co_change.handle_target import HandleTarget
from common.util import leave_one_out
from data.delete_record import DeleteRecord
from data.data_divider import DataDivider
from data.id_mapped import IdMapped


# def rank(word_list, target_word):
#     for i in range(len(word_list)):
#         if word_list[i] == target_word:
#             return i + 1
#     return 0
#
#
# def get_top_k(word_list, k):
#     if len(word_list) < k:
#         return word_list
#     return word_list[0:k]


# def print_recall(k, micro_recall, macro_recall):
#     print('k %d | micro recall %.2f | macro recall %.2f'
#           % (k, micro_recall, macro_recall))


class LowFreqEval:
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
        self.lowFreqTarmaqResultDao = LowFreqTarmaqResultDao()
        self.lowFreqTarmaqContextsResultDao = LowFreqTarmaqContextsResultDao()
        self.co_change = None
        self.handleTarget = None
        self.dataStore = DataStore()
        self.load_data()
        self.deleteRecord = DeleteRecord(git_name, expected_validate_length)

        self.md_list = self.dataStore.get_module_data(git_name)
        self.method_map = self.dataStore.get_method_map(git_name)

        self.dataDivider = DataDivider(self.md_list, expected_validate_length, self.most_recent)
        self.idMapped = IdMapped(self.dataDivider.get_train_data(), self.method_map)

        self.all_id_to_word = self.idMapped.all_id_to_word
        self.validate_data_commit_hash_list = self.dataDivider.validate_data_commit_hash_list

    def load_data(self):
        self.handleTarget = HandleTarget(self.git_name, self.expected_validate_length, self.most_recent)
        git_name_false = self.git_name + '_false'
        if self.most_recent > 0:
            git_name_false += '_' + str(self.most_recent)
        self.co_change = get_co_change(git_name_false)
        self.co_change = self.handleTarget.filter(self.co_change, False)

    def getTopK(self, topKList, k, cur_commit_hash):
        result = []
        count = 0
        if len(set(topKList)) != len(topKList):
            print('error length', topKList)
        for i in range(len(topKList)):
            if count == k:
                break
            item = topKList[i]
            if item in self.all_id_to_word:
                word = self.all_id_to_word[item]
                if self.deleteRecord.detect_is_in_vocab(cur_commit_hash, word):
                    result.append(item)
                    count += 1
        return result

    def validate_low_freq(self, threshold):
        data_loader = DataLoader(
            git_name=self.git_name,
            expected_validate_length=self.expected_validate_length,
            most_recent=self.most_recent,
            dataStore=self.dataStore,
            deleteRecord=None
        )
        validate_data = data_loader.validate_data
        lowFreqTrace = LowFreqTrace(self.git_name, self.expected_validate_length, self.most_recent, threshold)
        print(len(self.co_change), self.expected_validate_length,
              'is same as', len(self.co_change) == self.expected_validate_length)
        start_time = time.time()

        for i in range(len(self.co_change)):
            transaction_md = validate_data[i]
            pair_list_md = leave_one_out(transaction_md)

            pair_list = self.co_change[i]['list']
            commit_th = i
            lowFreqTrace.trace(self.expected_validate_length - i)

            cur_commit_hash = self.validate_data_commit_hash_list[i]
            for i2 in range(len(pair_list)):
                pair = pair_list[i2]
                # print('pair', pair)
                contexts = pair['contexts']
                target = pair['target']
                len1 = len(pair['topKList'])
                topKList = self.getTopK(pair['topKList'], 20, cur_commit_hash)
                len2 = len(topKList)
                # print('test', len1, len2)
                pair['topKList'] = topKList

                # target_md, contexts_md is md id format
                target_md = pair_list_md[i2]['target']
                contexts_md = pair_list_md[i2]['contexts']

                # if target != target_md:
                #     print('error', target, target_md)
                if len(contexts) != len(contexts_md):
                    print('contexts error', contexts, contexts_md)

                contexts_component_result = lowFreqTrace.contexts_component(contexts_md)
                lowFreqTrace.save_contexts_component_predict({
                    'contexts_component': contexts_component_result,
                    'predict_result': pair
                })
                if contexts_component_result['new_rate'] >= 1 and len(topKList) > 0:
                    print('error', contexts_md, topKList)
                self.lowFreqTarmaqContextsResultDao.insert({
                    'predict_result': pair,
                    'commit_th': commit_th,
                    'git_name': self.git_name,
                    'cur_expected_validate_length': self.expected_validate_length - i,
                    'most_recent': self.most_recent,
                    'version': self.version + self.sub_version,
                    'new_rate': str(contexts_component_result['new_rate']),
                    'new_list': contexts_component_result['new_list'],
                    'old_rate': str(contexts_component_result['old_rate']),
                    'old_list': contexts_component_result['old_list'],
                    'low_freq_rate': str(contexts_component_result['low_freq_rate']),
                    'low_freq_list': contexts_component_result['low_freq_list'],
                })
                if lowFreqTrace.is_low_freq_md_id(target):
                    lowFreqTrace.save_target_low_freq_predict({
                        'target': target,
                        'top100_aq': topKList
                    })
                    self.lowFreqTarmaqResultDao.insert({
                        'predict_result': pair,
                        'commit_th': commit_th,
                        'git_name': self.git_name,
                        'cur_expected_validate_length': self.expected_validate_length - i,
                        'most_recent': self.most_recent,
                        'version': self.version + self.sub_version,
                    })

            if i % 100 == 0:
                print('end with:', i, 'cost time:', time.time() - start_time)
        micro_recall = lowFreqTrace.target_low_freq_summary(self.k_list)
        print('tarmaq low freq file micro recall', micro_recall)


if __name__ == '__main__':
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    # git_name_list = ['tomcat']
    # first 3
    threshold_list = [3]
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        for i3 in range(len(threshold_list)):
            threshold = threshold_list[i3]
            version = git_name
            # sub_version = '_1_20_paper_' + str(threshold)
            sub_version = '_1_30_fix_final_' + str(threshold)
            eval = LowFreqEval(git_name, git_name, sub_version, 1000, 5000)
            eval.validate_low_freq(threshold)


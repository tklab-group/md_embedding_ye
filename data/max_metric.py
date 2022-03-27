import sys
sys.path.append('../')
import numpy as np
from config.config_default import get_config
from statistics import mean, median, stdev, variance
from data.data_divider import DataDivider
import time
from model.metric import Metric
from common.util import leave_one_out


class MaxMetric:
    def __init__(self,
                 dataDivider: DataDivider,
                 is_fix):
        self.dataDivider = dataDivider
        self.train_data = self.dataDivider.get_train_data()
        self.validate_data = self.dataDivider.get_validate_data()
        self.metric = Metric()

        self.is_fix = is_fix

        self.md_set = set()
        self.build_md_set()

        self.validate_contexts_target = []
        self.build_validate_contexts_target()

    def is_exist(self, md_id):
        return md_id in self.md_set

    def build_validate_contexts_target(self):
        for i in range(len(self.validate_data)):
            transaction = self.validate_data[i]
            pair_list = leave_one_out(transaction)
            temp_pair_list = []
            for j in range(len(pair_list)):
                pair = pair_list[j]
                target = pair['target']
                contexts = pair['contexts']

                temp_target = 1
                temp_contexts = 1
                temp_contexts_count = 0
                if not self.is_exist(target):
                    temp_target = -1
                for k in range(len(contexts)):
                    if self.is_exist(contexts[k]):
                        temp_contexts_count += 1
                if temp_contexts_count == 0:
                    temp_contexts = -1
                temp_pair_list.append({
                    'target': temp_target,
                    'contexts': temp_contexts
                })
            self.validate_contexts_target.append(temp_pair_list)
            if not self.is_fix:
                self.update_md_set(transaction)

    def build_md_set(self):
        for i in range(len(self.train_data)):
            train_data_item = self.train_data[i]
            for j in range(len(train_data_item)):
                self.md_set.add(train_data_item[j])

    def update_md_set(self, validate_data_item):
        for i in range(len(validate_data_item)):
            self.md_set.add(validate_data_item[i])

    def eval(self, is_consider_new_file=True):
        for i in range(len(self.validate_contexts_target)):
            pair_list = self.validate_contexts_target[i]
            for j in range(len(pair_list)):
                target = pair_list[j]['target']
                contexts = pair_list[j]['contexts']
                is_target_in_train = True
                if contexts == -1:
                    rec_i_c_len = 0
                    rank_i_c = 0
                else:
                    rec_i_c_len = 1
                    rank_i_c = 1
                if target == -1:
                    rank_i_c = 0
                    is_target_in_train = False
                commit_th = i
                # if not is_target_in_train:
                #     print(pair_list[j])
                self.metric.eval_with_commit(commit_th, rank_i_c, rec_i_c_len, is_target_in_train)
        # micro_recall, macro_recall, mrr, f_mrr = self.metric.summary(is_consider_new_file)
        # print('max metric is_fix %d | new file %d | micro recall %.2f | macro recall %.2f | mrr %.2f | f_mrr %.2f'
        #       % (self.is_fix, is_consider_new_file, micro_recall, macro_recall, mrr, f_mrr))
        micro_recall, macro_recall = self.metric.summary(is_consider_new_file)
        print('max metric is_fix %d | new file %d | micro recall %.2f | macro recall %.2f'
              % (self.is_fix, is_consider_new_file, micro_recall, macro_recall))


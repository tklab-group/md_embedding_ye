import sys
sys.path.append('../')
import pickle
import numpy as np
import heapq
from model.metric import Metric
import time
from data.data_store import DataStore
from co_change.handle_target import HandleTarget
from data.util import get_co_change
from data.data_divider import DataDivider
from data.id_mapped import IdMapped
from data.delete_record import DeleteRecord


class Evaluation:
    def __init__(self, git_name, expected_validate_length, most_recent, is_fix):
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent
        self.is_fix = is_fix

        self.k_list = [1, 5, 10, 15, 20]
        self.handleTarget = HandleTarget(git_name, expected_validate_length, most_recent)

        if is_fix:
            git_name_fix = git_name + '_true'
        else:
            git_name_fix = git_name + '_false'
        if most_recent > 0:
            git_name_fix += '_' + str(most_recent)
        print('git_name_fix', git_name_fix)
        self.co_change_list = get_co_change(git_name_fix)
        print(len(self.co_change_list))
        self.co_change_list = self.handleTarget.filter(self.co_change_list, is_fix)

        self.dataStore = DataStore()
        self.md_list = self.dataStore.get_module_data(git_name)
        self.method_map = self.dataStore.get_method_map(git_name)
        self.delete_record_list = self.dataStore.get_delete_record(git_name)

        self.dataDivider = DataDivider(self.md_list, expected_validate_length, self.most_recent)
        self.idMapped = IdMapped(self.dataDivider.get_train_data(), self.method_map)
        self.deleteRecord = DeleteRecord(git_name, expected_validate_length)

        self.all_id_to_word = self.idMapped.all_id_to_word
        self.validate_data = self.dataDivider.get_validate_data()
        self.validate_data_commit_hash_list = self.dataDivider.validate_data_commit_hash_list

    def rank(self, aq, target):
        if target == -1:
            return 0
        for i in range(len(aq)):
            if aq[i] == target:
                return i + 1
        return 0

    def getTopK(self, topKList, k, target, cur_commit_hash, is_delete_deleted_element=True):
        if not is_delete_deleted_element:
            if len(topKList) <= k:
                return topKList
            return topKList[0: k]
        else:
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
                    else:
                        if item == target:
                            print('error delete', target)
                            print('error delete topKList', topKList)
            return result

    def validate(self, k, is_consider_new_file=True, is_delete_deleted_element=True):
        metric = Metric()
        for i in range(len(self.co_change_list)):
            pair_list = self.co_change_list[i]['list']
            commit_th = i
            commit_th_hit_count = 0
            cur_commit_hash = self.validate_data_commit_hash_list[i]
            for j in range(len(pair_list)):
                pair = pair_list[j]
                # contexts = pair['contexts']
                target = pair['target']
                topKList = pair['topKList']
                if not target == -1:
                    is_target_in_train = True
                else:
                    # print('target not in train', pair_list)
                    is_target_in_train = False
                aq = self.getTopK(topKList, k, target, cur_commit_hash, is_delete_deleted_element)
                rec_i_c_len = len(aq)
                rank_i_c = self.rank(aq, target)
                if rank_i_c > 0:
                    commit_th_hit_count += 1
                metric.eval_with_commit(commit_th, rank_i_c, rec_i_c_len, is_target_in_train)

            # if is_delete_deleted_element:
            #     if self.is_fix:
            #         transaction = self.validate_data[i]
            #         cur_commit_hash = self.validate_data_commit_hash_list[i]
            #         self.deleteRecord.add_transaction(transaction, cur_commit_hash)
            #     else:
            #         cur_expected_validate_length = self.expected_validate_length - i
            #         self.build_delete_record_from(cur_expected_validate_length)
            # print(metric.summary())
            # if i % 100 == 0:
            #     print(k, i)
        return metric.summary(is_consider_new_file)

    def summary(self, is_consider_new_file=True):
        print('is consider new file', is_consider_new_file)
        for i in range(len(self.k_list)):
            k = self.k_list[i]
            micro_recall, macro_recall = self.validate(k, is_consider_new_file)
            # '1': {
            #     'micro': 0.251,
            #     'macro': 0.204
            # },
            print('\''+str(k)+'\': {', '\'micro\':', round(micro_recall, 3), ',', '\'macro\':', round(macro_recall, 3), '}')
            # print('k %d | micro recall %.3f | macro recall %.3f'
            #       % (k, micro_recall, macro_recall))

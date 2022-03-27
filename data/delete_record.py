import sys

sys.path.append('../')
import numpy as np
from config.config_default import get_config
from data.vocabulary import Vocabulary
from statistics import mean, median, stdev, variance
from data.mode_enum import Mode
from data.contexts_target_builder import ContextsTargetBuilder
from data.data_divider import DataDivider
from data.id_mapped import IdMapped
from data.util import get_module_data, get_method_map, save_module_method_map_pkl, load_module_method_map_pkl
from data.freq_counter import FreqCounter
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.sub_sampling import SubSampling
from data.negative_sampling import NegativeSampling
from data.max_metric import MaxMetric
import time
from data.data_store import DataStore
from data.pre_process import PreProcess
from data.rename_chain import RenameChain
import copy


class DeleteRecord:
    def __init__(self, git_name, expected_validate_length):
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length

        self.dataStore = DataStore()
        self.md_list = self.dataStore.get_module_data(git_name)
        self.method_map = self.dataStore.get_method_map(git_name)
        self.delete_record_list = self.dataStore.get_delete_record(git_name)

        self.dataDivider = DataDivider(self.md_list, self.expected_validate_length, 0)
        self.train_data = self.dataDivider.train_data
        # print('train_data len', len(self.train_data))
        self.train_data_commit_hash_list = self.dataDivider.train_data_commit_hash_list
        self.validate_data = self.dataDivider.validate_data
        # print('validate_data len', len(self.validate_data))
        self.validate_data_commit_hash_list = self.dataDivider.validate_data_commit_hash_list

        self.idMapped = IdMapped(self.dataDivider.get_train_data(), self.method_map)
        self.all_id_to_word = self.idMapped.all_id_to_word

        # commit_hash -> delete word set
        self.delete_record_map = {}
        self.simply_delete_record_list()

        # merge commit_hash_list and data_list
        commit_hash_list = []
        data_list = []
        for i in range(len(self.train_data_commit_hash_list)):
            commit_hash_list.append(self.train_data_commit_hash_list[i])
            data_list.append(self.train_data[i])
        for i in range(len(self.validate_data_commit_hash_list)):
            commit_hash_list.append(self.validate_data_commit_hash_list[i])
            data_list.append(self.validate_data[i])
        self.commit_hash_list = commit_hash_list
        self.data_list = data_list

        self.max_deleted_set_count = 0

        self.commit_hash_to_deleted_set = {}
        self.build_deleted_set()

        # print('max_deleted_set_count', self.max_deleted_set_count)

    def simply_delete_record_list(self):
        delete_record_map = {}
        count = 0
        for i in range(len(self.delete_record_list)):
            item = self.delete_record_list[i]
            fileNo = item['fileNo']
            commitId = item['commitId']
            # print('delete_record_list', fileNo, commitId)
            # all_id_to_wordと関係ない部分を削除
            if fileNo in self.all_id_to_word:
                count += 1
                # print('has relation', fileNo)
                word = self.all_id_to_word[fileNo]
                if commitId in delete_record_map:
                    delete_record_map[commitId].add(word)
                else:
                    delete_record_map[commitId] = set()
                    delete_record_map[commitId].add(word)
        self.delete_record_map = delete_record_map
        # print('has relation count', count)

    def get_word_list(self, transaction):
        word_list = []
        for i in range(len(transaction)):
            item = transaction[i]
            if item in self.all_id_to_word:
                word_list.append(self.all_id_to_word[item])
        return word_list

    def build_deleted_set(self):
        # commit hash -> deleted word set
        commit_hash_to_deleted_set = {}
        for i in range(len(self.commit_hash_list)):
            commit_hash = self.commit_hash_list[i]
            transaction = self.data_list[i]
            word_list = self.get_word_list(transaction)

            if i > 0:
                last_commit_hash = self.commit_hash_list[i - 1]
            else:
                last_commit_hash = None
            if last_commit_hash in commit_hash_to_deleted_set:
                last_deleted_set = copy.deepcopy(commit_hash_to_deleted_set[last_commit_hash])
            else:
                last_deleted_set = set()

            if commit_hash in self.delete_record_map:
                cur_commit_hash_deleted_set = self.delete_record_map[commit_hash]
            else:
                cur_commit_hash_deleted_set = set()
            # print('delete_record_map', commit_hash, cur_commit_hash_deleted_set)
            # last_deleted_setを追加
            for deleted_word in cur_commit_hash_deleted_set:
                last_deleted_set.add(deleted_word)

            # 今出現した単語をDelete listから削除
            cur_deleted_set = last_deleted_set - set(word_list)
            commit_hash_to_deleted_set[commit_hash] = cur_deleted_set
            self.max_deleted_set_count = max(self.max_deleted_set_count, len(cur_deleted_set))
        self.commit_hash_to_deleted_set = commit_hash_to_deleted_set

    def detect_is_in_vocab(self, cur_commit_hash, detect_word):
        if cur_commit_hash in self.commit_hash_to_deleted_set:
            deleted_set = self.commit_hash_to_deleted_set[cur_commit_hash]
            return detect_word not in deleted_set
        else:
            print('have no deleted set')
            return True

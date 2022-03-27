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
from data.delete_record import DeleteRecord


if __name__ == '__main__':
    git_name = 'tomcat'
    expected_validate_length = 1000
    most_recent = 5000

    dataStore = DataStore()
    md_list = dataStore.get_module_data(git_name)
    method_map = dataStore.get_method_map(git_name)
    delete_record_list = dataStore.get_delete_record(git_name)

    dataDivider = DataDivider(md_list, expected_validate_length, most_recent)
    idMapped = IdMapped(dataDivider.get_train_data(), method_map)
    deleteRecord = DeleteRecord(dataDivider, delete_record_list, idMapped)

    all_id_to_word = idMapped.all_id_to_word
    validate_data = dataDivider.get_validate_data()
    validate_data_commit_hash_list = dataDivider.validate_data_commit_hash_list
    train_data_commit_hash_list = dataDivider.train_data_commit_hash_list

    target_delete_count = 0
    target_delete_commit_count = 0
    for i in range(len(validate_data)):
        transaction = validate_data[i]
        cur_commit_hash = validate_data_commit_hash_list[i]
        if i > 0:
            last_commit_hash = validate_data_commit_hash_list[i - 1]
        else:
            last_commit_hash = train_data_commit_hash_list[len(train_data_commit_hash_list) - 1]
        # cur_expected_validate_length = expected_validate_length - i
        # dataDivider = DataDivider(md_list, cur_expected_validate_length, most_recent)
        # cur_train_data = dataDivider.get_train_data()
        # cur_validate_data = dataDivider.get_validate_data()
        # print('transaction', transaction)
        # print('last train data', cur_train_data[len(cur_train_data) - 1])
        # print('first validate data', cur_validate_data[0])
        # idMapped = IdMapped(cur_train_data, method_map)
        # deleteRecord = DeleteRecord(dataDivider, delete_record_list, idMapped)
        # deleteRecord.debug()
        have_error_delete_count = 0
        for j in range(len(transaction)):
            item = transaction[j]
            word = all_id_to_word[item]
            if not deleteRecord.detect_is_in_vocab(cur_commit_hash, word):
                target_delete_count += 1
                have_error_delete_count += 1
                print('error target delete', item, word)
        if have_error_delete_count > 0:
            target_delete_commit_count += 1
        print('end with', i)
        print()
    print('target_delete_commit_count', target_delete_commit_count)

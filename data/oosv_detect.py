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
from data.data_loader import DataLoader
from data.mongo import LowFreqContextsResultDao
from data.low_freq_trace import LowFreqTrace
from data.git_name_version import get_git_name_version
from common.util import save_data, load_data
import os


class OOSVDetect:
    def __init__(self):
        # cur_expected_validate_length -> dataLoader
        self.dataLoaderCache = {}

    def detect_for_low_freq_trace(self, dataStore, low_freq_trace_item):
        git_name = low_freq_trace_item['git_name']
        cur_expected_validate_length = low_freq_trace_item['cur_expected_validate_length']
        most_recent = low_freq_trace_item['most_recent']
        commit_hash = low_freq_trace_item['commit_hash']
        new_list = low_freq_trace_item['new_list']
        old_list = low_freq_trace_item['old_list']
        low_freq_list = low_freq_trace_item['low_freq_list']
        contexts = [] # new_list + old_list + low_freq_list
        for i in range(len(new_list)):
            contexts.append(new_list[i])
        for i in range(len(old_list)):
            contexts.append(old_list[i])
        for i in range(len(low_freq_list)):
            contexts.append(low_freq_list[i])
        # deleteRecord = DeleteRecord(git_name, expected_validate_length)
        if cur_expected_validate_length in self.dataLoaderCache:
            dataLoader = self.dataLoaderCache[cur_expected_validate_length]
        else:
            dataLoader = DataLoader(git_name=git_name,
                                    expected_validate_length=cur_expected_validate_length,
                                    most_recent=most_recent,
                                    dataStore=dataStore,
                                    deleteRecord=None,
                                    mode=Mode.SUB_WORD,
                                    )
            # cache
            self.dataLoaderCache[cur_expected_validate_length] = dataLoader
        vocab = dataLoader.vocab
        sub_word_to_index = vocab.sub_word_to_index
        # train_id_to_word = dataLoader.idMapped.train_id_to_word
        preProcess = dataLoader.preProcess
        renameChain = dataLoader.renameChain
        oosv_count = 0
        contexts_sub_word_list = []
        for i in range(len(contexts)):
            item = contexts[i]
            cur_name = renameChain.get_cur_name_by_hash(commit_hash, int(item))
            sub_word_list = preProcess.get_module_data_sub_word(cur_name)
            # contexts_sub_word_list += sub_word_list
            for j in range(len(sub_word_list)):
                contexts_sub_word_list.append(sub_word_list[j])
            # sub_word_set = set(sub_word_list)
            # for word in sub_word_set:
            #     if word not in sub_word_to_index:
            #         oosv_count += 1
        contexts_sub_word_set = set(contexts_sub_word_list)
        for word in contexts_sub_word_set:
            if word not in sub_word_to_index:
                oosv_count += 1
        total_count = len(contexts_sub_word_set)
        # print(oosv_count, total_count, round(oosv_count / total_count, 3))
        # setの割合にする
        return oosv_count, total_count


def single_main(dataStore, git_name, base_subword_version):
    print('single main', git_name, base_subword_version)
    detect = OOSVDetect()
    expected_validate_length = 1000
    most_recent = 5000
    threshold = 3
    lowFreqContextsResultDao = LowFreqContextsResultDao()
    cbow_list = lowFreqContextsResultDao.query_by(git_name, base_subword_version + str(threshold))
    # cbowLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
    my_list = []
    for doc in cbow_list:
        # cbowLowFreqTrace.save_contexts_component_predict({
        #     'contexts_component': {
        #         'new_rate': doc['new_rate'],
        #         'new_list': doc['new_list'],
        #         'old_rate': doc['old_rate'],
        #         'old_list': doc['old_list'],
        #         'low_freq_rate': doc['low_freq_rate'],
        #         'low_freq_list': doc['low_freq_list'],
        #     },
        #     'predict_result': doc['predict_result'],
        #     'commit_th': doc['commit_th']
        # })
        # 特定の条件に絞る
        predict_result = doc['predict_result']
        if int(predict_result['target']) != -1 and float(doc['new_rate']) >= 1:
            my_list.append(doc)
    print(len(my_list))
    oosv_count_list = []
    total_count_list = []
    for i in range(len(my_list)):
        start_time = time.time()
        oosv_count, total_count = detect.detect_for_low_freq_trace(dataStore, my_list[i])
        oosv_count_list.append(oosv_count)
        total_count_list.append(total_count)
        save_result(oosv_count_list, git_name + '_oosv_count_list')
        save_result(total_count_list, git_name + '_total_count_list')
        print('cost time', time.time() - start_time, i)
    # mean
    average_list = []
    for i in range(len(oosv_count_list)):
        average_list.append(oosv_count_list[i] / total_count_list[i])
    save_result(mean(average_list), git_name + '_mean_average_list')
    print('mean result', git_name, mean(average_list))
    print('end', git_name, base_subword_version)
    del detect.dataLoaderCache
    del detect


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


def main(dataStore):
    # git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    git_name_list = ['tomcat', 'lucene', 'hbase', 'cassandra']
    sub_version = '_1_20_paper_'
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        subword_version_list = get_git_name_version(git_name, 'subword')
        base_subword_version = subword_version_list[0]['version'] + sub_version
        single_main(dataStore, git_name, base_subword_version)


if __name__ == '__main__':
    # print(load_result('tomcattotal_count_list'))
    dataStore = DataStore()
    main(dataStore)

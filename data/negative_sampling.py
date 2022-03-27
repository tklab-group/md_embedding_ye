import sys
sys.path.append('../')
import torch
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.id_mapped import IdMapped
import numpy as np
from common.util import leave_one_out
from config.config_default import get_config
from data.mode_enum import Mode
from data.sub_sampling import SubSampling
from common.util import to_sub_word, n_gram_from_sub_word_set
from collections import Counter
from statistics import mean, median, stdev, variance
from data.freq_counter import FreqCounter
import copy
import time


class NegativeSampling:
    def __init__(self,
                 freqCounter: FreqCounter,
                 idMapped: IdMapped,
                 train_data):
        config_all = get_config()
        self.torch_seed = config_all['torch_seed']
        self.negative_sampling_num = config_all['negative_sampling_num']
        self.padding_id = config_all['dataset']['padding_id']

        self.train_data = train_data
        self.freqCounter = freqCounter
        self.idMapped = idMapped
        self.word_list = []
        self.freq_list = []

        self.md_id_to_position_index = {}

        self.build_freq_time = time.time()
        self.build_freq()
        self.build_freq_time = time.time() - self.build_freq_time

        self.build_position_index_time = time.time()
        self.build_position_index()
        self.build_position_index_time = time.time() - self.build_position_index_time

    def debug_info(self):
        print('build_freq_time', self.build_freq_time)
        print('build_position_index_time', self.build_position_index_time)

    def build_freq(self):
        counter = self.freqCounter.word_counter
        total_count = self.freqCounter.word_total_count
        for word, count in counter.items():
            self.word_list.append(word)
            self.freq_list.append((count / total_count) ** (3. / 4.))
            # self.freq_list.append((count / total_count))
        self.freq_list = torch.from_numpy(np.array(self.freq_list))

    def build_position_index(self):
        for i in range(len(self.train_data)):
            train_data_item = self.train_data[i]
            for j in range(len(train_data_item)):
                item = train_data_item[j]
                if item not in self.md_id_to_position_index:
                    self.md_id_to_position_index[item] = set()
                self.md_id_to_position_index[item].add(i)

    def is_co_change(self, query, md_id):
        # query and md_id are md id
        if md_id not in self.md_id_to_position_index:
            # print('is_co_change first end')
            return False
        md_id_position_index_set = self.md_id_to_position_index[md_id]
        # print(md_id_position_index_set)
        query_position_set = set()
        for i in range(len(query)):
            item = query[i]
            query_position_set = query_position_set.union(self.md_id_to_position_index[item])
        # print('is_co_change', query_position_set)
        if len(md_id_position_index_set & query_position_set) > 0:
            return True
        return False

    def _sampling(self):
        # return sampling word list
        result = []
        if len(self.freq_list) < self.negative_sampling_num:
            return result
        # torch.manual_seed(self.torch_seed)
        idx = torch.multinomial(self.freq_list, self.negative_sampling_num, replacement=True)
        for i in range(len(idx)):
            result.append(self.word_list[idx[i].numpy()])
        return result

    def sampling(self, contexts, target):
        # return [0, 0, 0, 0, 0]
        # print('sampling', contexts, target)
        # contexts and target are md id
        word_to_id = self.idMapped.train_word_to_id

        negative_md_ids = []
        negative_word_list = self._sampling()
        for i in range(len(negative_word_list)):
            negative_md_ids.append(word_to_id[negative_word_list[i]])
        # return negative_md_ids

        # temp_contexts = copy.deepcopy(contexts)
        # print('temp_contexts', temp_contexts)
        # temp_contexts.append(target)
        # query = temp_contexts
        query = np.append(contexts, target)
        # print('query', contexts, target, query)

        result = []
        for i in range(len(negative_md_ids)):
            negative_md_id = negative_md_ids[i]
            if not self.is_co_change(query, negative_md_id):
                result.append(negative_md_id)
        return result





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


def is_hit(target, top100_aq, k):
    k = min(k, len(top100_aq))
    if target == -1:
        return False
    for i in range(len(top100_aq)):
        if i + 1 > k:
            break
        if target == top100_aq[i]:
            return True
    return False


class LowFreqTrace:
    # 元々はTarmaqはあんまり変更されていない要素に対して推薦が弱いと思ったので
    # あんまり変更されていない要素は過去のコミットにおいて出現回数が1回から3回までの要素
    # 今は直近5000の評価モードしか使えない
    def __init__(self, git_name, expected_validate_length, most_recent, threshold):
        print('low freq trace param', git_name, expected_validate_length, most_recent, threshold)
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent

        self.dataStore = DataStore()
        self.md_list = self.dataStore.get_module_data(git_name)
        self.method_map = self.dataStore.get_method_map(git_name)

        self.idMapped = IdMapped([], self.method_map)
        self.all_id_to_word = self.idMapped.all_id_to_word

        # 出現頻度を記録
        self.word_counter = {}
        self.word_total_count = 0
        self.vocab_set = set()

        self.threshold = threshold

        # config
        config_all = get_config()
        self.config_all = config_all
        self.padding_word = config_all['dataset']['padding_word']

        # targetがあんまり変更されていないの予測結果リスト
        self.target_low_freq_predict_list = []

        # Contextsの各構成とその予測結果リスト
        self.contexts_component_predict_list = []
        self.new_rate_list = []
        self.old_rate_list = []
        self.low_freq_rate_list = []

    def get_word_from(self, md_id):
        if md_id in self.all_id_to_word:
            return self.all_id_to_word[md_id]
        # print('get word from 例外', md_id)
        return self.padding_word

    def get_word_list_from(self, transaction):
        word_list = []
        for i in range(len(transaction)):
            md_id = transaction[i]
            word_list.append(self.get_word_from(md_id))
        return word_list

    def is_new_word(self, word):
        if word not in self.vocab_set:
            return True
        return False

    def trace(self, cur_expected_validate_length):
        # transaction md_id format
        dataDivider = DataDivider(self.md_list, cur_expected_validate_length, self.most_recent)
        train_data = dataDivider.get_train_data()
        idMapped = IdMapped(train_data, self.method_map)
        train_words = idMapped.train_words
        train_id_to_word = idMapped.train_id_to_word

        word_counter = {}
        word_total_count = 0
        for i in range(len(train_data)):
            id_list = train_data[i]
            word_total_count += len(id_list)
            for j in range(len(id_list)):
                md_id = id_list[j]
                word = train_id_to_word[md_id]
                # count word freq
                if word in word_counter:
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1
        self.word_counter = word_counter
        self.word_total_count = word_total_count
        self.vocab_set = train_words

    def is_low_freq_md_id(self, md_id):
        if md_id == -1:
            return False
        word = self.get_word_from(md_id)
        return self.is_low_freq_word(word)

    def is_low_freq_word(self, word):
        if word == self.padding_word:
            # print('is_low_freq_word is padding word')
            return False
        if word in self.word_counter:
            count = self.word_counter[word]
            if count <= self.threshold:
                return True
        return False

    def contexts_component(self, contexts_md):
        total_len = len(contexts_md)
        if total_len == 0:
            return {
                'new_rate': 0,
                'new_list': [],
                'old_rate': 0,
                'old_list': [],
                'low_freq_rate': 0,
                'low_freq_list': [],
            }
        new_file_list = []
        low_freq_file_list = []
        old_file_list = []
        for i in range(len(contexts_md)):
            md_id = contexts_md[i]
            word = self.all_id_to_word[md_id]
            if self.is_new_word(word):
                new_file_list.append(str(md_id))
            else:
                if self.is_low_freq_word(word):
                    low_freq_file_list.append(str(md_id))
                else:
                    old_file_list.append(str(md_id))
        new_rate = len(new_file_list) / total_len
        low_freq_rate = len(low_freq_file_list) / total_len
        old_rate = len(old_file_list) / total_len
        return {
            'new_rate': new_rate,
            'new_list': new_file_list,
            'old_rate': old_rate,
            'old_list': old_file_list,
            'low_freq_rate': low_freq_rate,
            'low_freq_list': low_freq_file_list
        }

    def save_contexts_component_predict(self, predict_result):
        self.contexts_component_predict_list.append(predict_result)

    def stat_contexts_component(self):
        print('new mean %f | median %f | stdev %f | variance %f'
              % (mean(self.new_rate_list), median(self.new_rate_list), stdev(self.new_rate_list),
                 variance(self.new_rate_list)))
        print('old mean %f | median %f | stdev %f | variance %f'
              % (mean(self.old_rate_list), median(self.old_rate_list), stdev(self.old_rate_list),
                 variance(self.old_rate_list)))
        print('low_freq mean %f | median %f | stdev %f | variance %f'
              % (mean(self.low_freq_rate_list), median(self.low_freq_rate_list), stdev(self.low_freq_rate_list),
                 variance(self.low_freq_rate_list)))

    def contexts_component_summary(self, k_list, over_threshold=0.5):
        new_micro_recall = {}
        old_micro_recall = {}
        low_freq_micro_recall = {}
        # targetが−1の場合は除外してMirco＿recallを評価する
        # over_threshold = 0.5
        for i in range(len(k_list)):
            k = k_list[i]
            old_over_hit = 0
            old_over_total = 0
            new_over_hit = 0
            new_over_total = 0
            low_freq_over_hit = 0
            low_freq_over_total = 0
            target_min_one_count = 0
            target_min_one_commit_th_list = []
            for i2 in range(len(self.contexts_component_predict_list)):
                item = self.contexts_component_predict_list[i2]
                contexts_component = item['contexts_component']
                new_rate = float(contexts_component['new_rate'])
                # new_list = contexts_component['new_list']
                old_rate = float(contexts_component['old_rate'])
                # old_list = contexts_component['old_list']
                low_freq_rate = float(contexts_component['low_freq_rate'])
                # low_freq_list = contexts_component['low_freq_list']

                predict_result = item['predict_result']
                target = predict_result['target']
                top100_aq = predict_result['top100_aq']

                # targetが−1の場合は除外
                if target == -1:
                    target_min_one_count += 1
                    target_min_one_commit_th_list.append(item['commit_th'])
                    continue

                self.new_rate_list.append(new_rate)
                self.old_rate_list.append(old_rate)
                self.low_freq_rate_list.append(low_freq_rate)
                # print('new_rate', new_rate)
                # print('old_rate', old_rate)
                # print('low_freq_rate', low_freq_rate)
                # print('total', new_rate + old_rate + low_freq_rate)
                # if new_rate >= over_threshold:
                #     print('target', target, k)
                #     print('top100_aq', top100_aq)
                cur_is_hit = is_hit(target, top100_aq, k)
                if new_rate >= over_threshold:
                    new_over_total += 1
                    if cur_is_hit:
                        new_over_hit += 1
                if old_rate >= over_threshold:
                    old_over_total += 1
                    if cur_is_hit:
                        old_over_hit += 1
                if low_freq_rate >= over_threshold:
                    low_freq_over_total += 1
                    if cur_is_hit:
                        low_freq_over_hit += 1
            new_micro_recall[k] = (new_over_hit, new_over_total, new_over_hit / new_over_total)
            old_micro_recall[k] = (old_over_hit, old_over_total, old_over_hit / old_over_total)
            low_freq_micro_recall[k] = (low_freq_over_hit, low_freq_over_total, low_freq_over_hit / low_freq_over_total)
            # print('target_min_one_count', target_min_one_count, target_min_one_commit_th_list)
        # self.stat_contexts_component()
        return {
            'new': new_micro_recall,
            'old': old_micro_recall,
            'low_freq': low_freq_micro_recall,
        }

    def save_target_low_freq_predict(self, predict_result):
        # predict_result=>{'target', 'top100_aq',...}
        self.target_low_freq_predict_list.append(predict_result)

    def target_low_freq_summary(self, k_list):
        micro_recall = {}
        total_count = len(self.target_low_freq_predict_list)
        if total_count == 0:
            for i in range(len(k_list)):
                k = k_list[i]
                micro_recall[k] = 0
        else:
            for i in range(len(k_list)):
                k = k_list[i]
                hit_count = 0
                for ii in range(len(self.target_low_freq_predict_list)):
                    item = self.target_low_freq_predict_list[ii]
                    target = item['target']
                    top100_aq = item['top100_aq']
                    if is_hit(target, top100_aq, k):
                        hit_count += 1
                micro_recall[k] = hit_count / total_count
        return micro_recall





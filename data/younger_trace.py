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


class YoungerTrace:
    # 若いファイルとは、訓練データには出現していない、かつ評価データで１回から３回まで出現したファイルのことである
    # しかし、訓練データには出現していないことは過去で出現しなかったとは限らない
    def __init__(self, all_id_to_word):
        self.all_id_to_word = all_id_to_word
        # 出現頻度を記録
        self.freq = {}
        # threshold回以下のは若いファイル
        self.threshold = 3
        self.vocab_set = set()

        # config
        config_all = get_config()
        self.config_all = config_all
        self.padding_word = config_all['dataset']['padding_word']

        # targetが若いファイルの予測結果リスト
        self.target_younger_predict_list = []

        # Contextsの各構成とその予測結果リスト
        self.contexts_component_predict_list = []
        self.new_rate_list = []
        self.old_rate_list = []
        self.younger_rate_list = []

    def get_word_from(self, md_id):
        if md_id in self.all_id_to_word:
            return self.all_id_to_word[md_id]
        print('get word from 例外', md_id)
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

    def trace(self, vocab_list, transaction):
        # transaction md_id format
        self.vocab_set = set(vocab_list)
        word_list = self.get_word_list_from(transaction)
        for i in range(len(word_list)):
            word = word_list[i]
            if word in self.freq:
                self.freq[word] += 1
            else:
                if self.is_new_word(word):
                    self.freq[word] = 1

    def is_younger_md_id(self, md_id):
        # if md_id == -1:
        #     return False
        word = self.get_word_from(md_id)
        return self.is_younger_word(word)

    def is_younger_word(self, word):
        if word == self.padding_word:
            print('is_younger_word is padding word')
            return False
        if word in self.freq:
            count = self.freq[word]
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
                'younger_rate': 0,
                'younger_list': [],
            }
        new_file_list = []
        younger_file_list = []
        old_file_list = []
        for i in range(len(contexts_md)):
            md_id = contexts_md[i]
            if self.is_younger_md_id(md_id):
                younger_file_list.append(str(md_id))
            else:
                if md_id in self.all_id_to_word:
                    word = self.all_id_to_word[md_id]
                    if self.is_new_word(word):
                        new_file_list.append(str(md_id))
                    else:
                        old_file_list.append(str(md_id))
        new_rate = len(new_file_list) / total_len
        younger_rate = len(younger_file_list) / total_len
        old_rate = len(old_file_list) / total_len
        return {
            'new_rate': new_rate,
            'new_list': new_file_list,
            'old_rate': old_rate,
            'old_list': old_file_list,
            'younger_rate': younger_rate,
            'younger_list': younger_file_list
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
        print('younger mean %f | median %f | stdev %f | variance %f'
              % (mean(self.younger_rate_list), median(self.younger_rate_list), stdev(self.younger_rate_list),
                 variance(self.younger_rate_list)))

    def contexts_component_summary(self, k_list):
        new_micro_recall = {}
        old_micro_recall = {}
        younger_micro_recall = {}
        # targetが−1の場合は除外してMirco＿recallを評価する
        over_threshold = 0.5
        for i in range(len(k_list)):
            k = k_list[i]
            old_over_hit = 0
            old_over_total = 0
            new_over_hit = 0
            new_over_total = 0
            younger_over_hit = 0
            younger_over_total = 0
            target_min_one_count = 0
            target_min_one_commit_th_list = []
            for i2 in range(len(self.contexts_component_predict_list)):
                item = self.contexts_component_predict_list[i2]
                contexts_component = item['contexts_component']
                new_rate = float(contexts_component['new_rate'])
                new_list = contexts_component['new_list']
                old_rate = float(contexts_component['old_rate'])
                old_list = contexts_component['old_list']
                younger_rate = float(contexts_component['younger_rate'])
                younger_list = contexts_component['younger_list']

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
                self.younger_rate_list.append(younger_rate)

                cur_is_hit = is_hit(target, top100_aq, k)
                if new_rate >= over_threshold:
                    new_over_total += 1
                    if cur_is_hit:
                        new_over_hit += 1
                if old_rate >= over_threshold:
                    old_over_total += 1
                    if cur_is_hit:
                        old_over_hit += 1
                if younger_rate >= over_threshold:
                    younger_over_total += 1
                    if cur_is_hit:
                        younger_over_hit += 1
            new_micro_recall[k] = (new_over_hit, new_over_total, new_over_hit / new_over_total)
            old_micro_recall[k] = (old_over_hit, old_over_total, old_over_hit / old_over_total)
            younger_micro_recall[k] = (younger_over_hit, younger_over_total, younger_over_hit / younger_over_total)
            print('target_min_one_count', target_min_one_count, target_min_one_commit_th_list)
        self.stat_contexts_component()
        return {
            'new': new_micro_recall,
            'old': old_micro_recall,
            'younger': younger_micro_recall,
        }

    def save_target_younger_predict(self, predict_result):
        # predict_result=>{'target', 'top100_aq',...}
        self.target_younger_predict_list.append(predict_result)

    def target_younger_summary(self, k_list):
        micro_recall = {}
        total_count = len(self.target_younger_predict_list)
        if total_count == 0:
            for i in range(len(k_list)):
                k = k_list[i]
                micro_recall[k] = 0
        else:
            for i in range(len(k_list)):
                k = k_list[i]
                hit_count = 0
                for ii in range(len(self.target_younger_predict_list)):
                    item = self.target_younger_predict_list[ii]
                    target = item['target']
                    top100_aq = item['top100_aq']
                    if is_hit(target, top100_aq, k):
                        hit_count += 1
                micro_recall[k] = hit_count / total_count
        return micro_recall

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


class EvaluationMixSubwordDataBase:
    def __init__(self,
                 git_name,
                 version_normal,
                 version_subword,
                 expected_validate_length,
                 ):
        self.git_name = git_name
        self.version_normal = version_normal
        self.version_subword = version_subword
        self.expected_validate_length = expected_validate_length
        self.dao = PredictDao()
        self.k_list = [1, 5, 10, 15, 20]
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()
        self.predict_list = []
        self.load_data()

    def load_data(self):
        query_list_normal = self.dao.query_by(self.git_name, self.version_normal, "NORMAL")
        query_list_subword = self.dao.query_by(self.git_name, self.version_subword, "SUB_WORD")
        list_normal = []
        list_normal_count = []
        list_subword = []
        list_subword_count = []
        for doc in query_list_normal:
            list_normal.append(doc)
            list_normal_count.append(len(doc['predict_result']))
        for doc in query_list_subword:
            list_subword.append(doc)
            list_subword_count.append(len(doc['predict_result']))

        # check
        print(len(list_normal), len(list_subword))
        not_match_count = 0
        for i in range(len(list_normal_count)):
            # print(i, list_normal_count[i], list_subword_count[i])
            if list_normal_count[i] != list_subword_count[i]:
                not_match_count += 1
                print(i, list_normal_count[i], list_subword_count[i])
        print('not_match_count', not_match_count)
        for j in range(len(list_normal)):
            list_normal_item = list_normal[j]
            list_subword_item = list_subword[j]

            predict_result = list_normal_item['predict_result']
            commit_th = list_normal_item['commit_th']
            commit_hash = list_normal_item['commit_hash']
            for i in range(len(predict_result)):
                item = predict_result[i]
                target = item['target']
                target_embedding_index = item['target_embedding_index']
                target_word = item['target_word']
                contexts = item['contexts']
                if len(contexts) > 0:
                    # negative_sampling = item['negative_sampling']
                    top100_aq = item['top100_aq']
                    top100_prob_list = item['top100_prob_list']
                else:
                    if len(list_subword_item['predict_result'][i]['contexts']) > 0:
                        # subword model try to predict
                        top100_aq = list_subword_item['predict_result'][i]['top100_aq']
                        top100_prob_list = list_subword_item['predict_result'][i]['top100_prob_list']
                    else:
                        top100_aq = []
                        top100_prob_list = []
                if target != -1:
                    is_target_in_train = True
                else:
                    is_target_in_train = False
                for ii in range(len(self.k_list)):
                    k = self.k_list[ii]
                    top_k = get_top_k(top100_aq, k)
                    rank_i_c = rank(top_k, target)
                    rec_i_c_len = len(top_k)
                    self.metric_list[k].eval_with_commit(
                        commit_th,
                        rank_i_c,
                        rec_i_c_len,
                        is_target_in_train
                    )

    def validate(self):
        for i in range(len(self.metric_list)):
            k = self.k_list[i]

            metric = self.metric_list[k]
            # is_consider_new_file=False
            micro_recall, macro_recall = metric.summary(True)
            print_recall(k, micro_recall, macro_recall)


if __name__ == '__main__':
    # not_match_count 780
    # print('tomcat normal dim:700 max_epoch:20 batch_size:512')
    # print('tomcat subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('tomcat', 'v1', '2021-12-23_tomcat_subword_like_paper', 1000)
    # eval.validate()

    # print('tomcat normal dim:700 max_epoch:20 batch_size:512')
    # print('tomcat subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('tomcat', '2021-12-24_tomcat_normal', '2021-12-23_tomcat_subword_like_paper', 1000)
    # eval.validate()

    # print('tomcat normal dim:700 max_epoch:20 batch_size:256 顺序')
    # print('tomcat subword dim:700 max_epoch:20 batch_size:128 顺序')
    # eval = EvaluationMixSubwordDataBase('tomcat', '2021_12_30_normal_tomcat_1', '2021_1_2_tsubame_sub_word_tomcat_1', 1000)
    # eval.validate()

    print('tomcat normal dim:700 max_epoch:10 batch_size:256 顺序')
    print('tomcat subword dim:700 max_epoch:20 batch_size:128 顺序')
    eval = EvaluationMixSubwordDataBase('tomcat', '2022_1_8_tomcat_tomcat_6', '2022_1_8_tomcat_subword_tomcat_6', 1000)
    eval.validate()

    # not_match_count 0
    # print('tomcat normal dim:700 max_epoch:20 batch_size:512')
    # print('tomcat subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('tomcat', 'v1', 'v2', 1000)
    # eval.validate()

    # not_match_count 0
    # print('pig normal dim:700 max_epoch:20 batch_size:512')
    # print('pig subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('pig', '2021-12-20', '2021-12-20', 1000)
    # eval.validate()

    # not_match_count 0
    # print('spark normal dim:700 max_epoch:20 batch_size:512')
    # print('spark subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('spark', '2021-12-20', '2021-12-20', 1000)
    # eval.validate()

    # not_match_count 0
    # print('struts normal dim:700 max_epoch:20 batch_size:512')
    # print('struts subword dim:700 max_epoch:10 batch_size:64')
    # eval = EvaluationMixSubwordDataBase('struts', '2021-12-20', '2021-12-20', 1000)
    # eval.validate()

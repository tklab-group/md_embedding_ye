import sys

sys.path.append('../')
import os
import torch
import time
import datetime
from data.data_loader import DataLoader
from model.trainer import Trainer
from model.cbow import EmbeddingModel
from model.eval import Evaluation
from data.mode_enum import Mode
from model.main import Main, print_recall
from model.multi_eval import MultiEvaluation
from data.data_store import DataStore
from data.delete_record import DeleteRecord
from data.util import save_predict_result, load_predict_result, is_predict_result_exits


class MultiMain:
    def __init__(self,
                 dataStore: DataStore,
                 deleteRecord: DeleteRecord,
                 version,
                 dim=100,
                 batch_size=32,
                 max_epoch=10,
                 git_name=None,
                 expected_validate_length=1000,
                 most_recent=5000,
                 validate_data=None,
                 mode=Mode.NORMAL,
                 device=None,
                 lr=1e-3,
                 is_sub_sampling=False,
                 is_subword_sub_sampling=False,
                 is_negative_sampling=False,
                 is_cosine_similarity_predict=False,
                 is_check_rename=True,
                 is_contexts_extend=False,
                 contribution_rate=0.5,
                 is_use_package=True,
                 is_use_class_name=True,
                 is_use_return_type=True,
                 is_use_method_name=True,
                 is_use_param_type=True,
                 is_use_param_name=True,
                 is_split_train_data=False,
                 is_simple_handle_package=False,
                 is_simple_handle_class_name=False,
                 is_simple_handle_return_type=False,
                 is_simple_handle_method_name=False,
                 is_simple_handle_param_type=False,
                 is_simple_handle_param_name=False,
                 is_predict_with_file_level=False,
                 is_mark_respective_type=False,
                 seed=6,
                 shuffle=False
                 ):
        self.dim = dim
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent
        self.mode = mode
        self.validate_data = validate_data
        self.device = device
        self.lr = lr
        self.is_sub_sampling = is_sub_sampling
        self.is_subword_sub_sampling = is_subword_sub_sampling
        self.is_negative_sampling = is_negative_sampling
        self.is_cosine_similarity_predict = is_cosine_similarity_predict
        self.is_check_rename = is_check_rename
        self.is_contexts_extend = is_contexts_extend
        self.contribution_rate = contribution_rate
        self.is_use_package = is_use_package
        self.is_use_class_name = is_use_class_name
        self.is_use_return_type = is_use_return_type
        self.is_use_method_name = is_use_method_name
        self.is_use_param_type = is_use_param_type
        self.is_use_param_name = is_use_param_name
        self.is_split_train_data = is_split_train_data
        self.is_simple_handle_package = is_simple_handle_package
        self.is_simple_handle_class_name = is_simple_handle_class_name
        self.is_simple_handle_return_type = is_simple_handle_return_type
        self.is_simple_handle_method_name = is_simple_handle_method_name
        self.is_simple_handle_param_type = is_simple_handle_param_type
        self.is_simple_handle_param_name = is_simple_handle_param_name
        self.is_predict_with_file_level = is_predict_with_file_level
        self.is_mark_respective_type = is_mark_respective_type
        self.seed = seed
        self.shuffle = shuffle

        self.version = version
        self.dataStore = dataStore
        self.deleteRecord = deleteRecord

        self.multiEvaluation = MultiEvaluation(
            version=version,
            is_save=True
        )
        self.multiEvaluationNew = MultiEvaluation(
            version=version,
            is_save=False
        )

    def train_and_eval(self):
        # 前回止められた状況を考慮する
        if is_predict_result_exits(self.git_name, self.version + '_temp'):
            print(self.git_name, self.version, 'temp exits')
            temp_predict_result = load_predict_result(self.git_name, self.version + '_temp')
            print('temp size', len(temp_predict_result))
        else:
            print(self.git_name, self.version, 'temp not exits')
            temp_predict_result = []
        completed_commit_th_set = set()
        for i in range(len(temp_predict_result)):
            item = temp_predict_result[i]
            completed_commit_th_set.add(item['commit_th'])
        print('completed_commit_th_set', len(completed_commit_th_set), completed_commit_th_set)
        if len(temp_predict_result) > 0:
            self.multiEvaluation.set_predict_result_list(temp_predict_result)
            print('set temp predict result', len(self.multiEvaluation.predict_result_list))
        for i in range(self.expected_validate_length):
            commit_th = i
            # 既に完成したcommit_thに対してスキップする
            if commit_th in completed_commit_th_set:
                continue
            start_time = time.time()
            transaction = self.validate_data[i]
            main = Main(git_name=self.git_name,
                        expected_validate_length=self.expected_validate_length - i,
                        most_recent=self.most_recent,
                        mode=self.mode,
                        max_epoch=self.max_epoch,
                        dim=self.dim,
                        batch_size=self.batch_size,
                        lr=self.lr,
                        is_sub_sampling=self.is_sub_sampling,
                        is_subword_sub_sampling=self.is_subword_sub_sampling,
                        is_negative_sampling=self.is_negative_sampling,
                        dataStore=self.dataStore,
                        deleteRecord=self.deleteRecord,
                        is_check_rename=self.is_check_rename,
                        is_cosine_similarity_predict=self.is_cosine_similarity_predict,
                        contribution_rate=self.contribution_rate,
                        is_fix=False,
                        is_use_package=self.is_use_package,
                        is_use_class_name=self.is_use_class_name,
                        is_use_return_type=self.is_use_return_type,
                        is_use_method_name=self.is_use_method_name,
                        is_use_param_type=self.is_use_param_type,
                        is_use_param_name=self.is_use_param_name,
                        is_split_train_data=self.is_split_train_data,
                        is_simple_handle_package=self.is_simple_handle_package,
                        is_simple_handle_class_name=self.is_simple_handle_class_name,
                        is_simple_handle_return_type=self.is_simple_handle_return_type,
                        is_simple_handle_method_name=self.is_simple_handle_method_name,
                        is_simple_handle_param_type=self.is_simple_handle_param_type,
                        is_simple_handle_param_name=self.is_simple_handle_param_name,
                        is_predict_with_file_level=self.is_predict_with_file_level,
                        is_mark_respective_type=self.is_mark_respective_type,
                        seed=self.seed,
                        shuffle=self.shuffle
                        )
            main.train(device=self.device)
            main.model.eval()
            # get current validate data hash
            commit_hash = main.data_loader.dataDivider.validate_data_commit_hash_list[0]
            # print(i, commit_hash, self.expected_validate_length)
            self.multiEvaluation.eval(transaction,
                                      i,
                                      commit_hash,
                                      main.model,
                                      main.data_loader,
                                      main,
                                      self.mode,
                                      self.is_negative_sampling,
                                      False)
            # self.multiEvaluationNew.eval(transaction,
            #                              i,
            #                              commit_hash,
            #                              main.model,
            #                              main.data_loader,
            #                              self.mode,
            #                              self.is_negative_sampling,
            #                              True)
            del main.model
            del main
            torch.cuda.empty_cache()
            print('cost time', i, time.time() - start_time)
            if i % 10 == 0:
                print('end with', i)

    def summary(self):
        k_list, metric_list = self.multiEvaluation.summary()
        # k_list_new, metric_list_new = self.multiEvaluationNew.summary()

        for i in range(len(metric_list)):
            k = k_list[i]

            metric = metric_list[k]
            # is_consider_new_file=False
            micro_recall, macro_recall = metric.summary(False)
            print_recall(k, False, False, micro_recall, macro_recall)
            # is_consider_new_file=True
            micro_recall, macro_recall = metric.summary(True)
            print_recall(k, True, False, micro_recall, macro_recall)

            # metricNew = metric_list_new[k]
            # # is_consider_new_file=False
            # micro_recall, macro_recall = metricNew.summary(False)
            # print_recall(k, False, True, micro_recall, macro_recall)
            # # is_consider_new_file=True
            # micro_recall, macro_recall = metricNew.summary(True)
            # print_recall(k, True, True, micro_recall, macro_recall)

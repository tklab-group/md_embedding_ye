import sys
sys.path.append('../')
import pickle
import numpy as np
import heapq
from model.metric import Metric
import time
import datetime
import torch
from config.config_default import get_config
from data.mode_enum import Mode
from model.eval import Evaluation
from data.mongo import PredictDao
from model.main import Main
from data.util import save_predict_result


class MultiEvaluation:
    def __init__(self,
                 version,
                 is_save=True):
        self.k_list = [1, 5, 10, 15, 20]
        self.version = version
        self.is_save = is_save
        self.metric_list = {}
        for i in range(len(self.k_list)):
            self.metric_list[self.k_list[i]] = Metric()
        self.dao = PredictDao()
        config = get_config()
        self.is_load_data_from_pkl = config['is_load_data_from_pkl']
        self.predict_result_list = []
        self.git_name = ''

    def set_predict_result_list(self, temp_predict_result_list):
        self.predict_result_list = temp_predict_result_list

    def eval(self,
             transaction,
             commit_th,
             commit_hash,
             model,
             data_loader,
             main: Main,
             mode=Mode.NORMAL,
             is_negative_sampling=False,
             is_predict_new_file=True):
        # print('eval', self.k, transaction, commit_th, model, data_loader, mode)
        evaluation = Evaluation(
            model,
            data_loader,
            mode,
            is_negative_sampling)
        transaction_result, pair_result = evaluation.validate_with_transaction(
            transaction,
            commit_th,
            is_predict_new_file,
            commit_hash
        )
        for ii in range(len(self.k_list)):
            k = self.k_list[ii]
            for iii in range(len(transaction)):
                self.metric_list[k].eval_with_commit(
                    transaction_result[k]['commit_th_list'][iii],
                    transaction_result[k]['rank_i_c_list'][iii],
                    transaction_result[k]['rec_i_c_len_list'][iii],
                    transaction_result[k]['is_target_in_train_list'][iii],
                )
        str_mode = 'NORMAL'
        if mode == Mode.NORMAL:
            str_mode = 'NORMAL'
        elif mode == Mode.SUB_WORD:
            str_mode = 'SUB_WORD'
        elif mode == Mode.SUB_WORD_NO_FULL:
            str_mode = 'SUB_WORD_NO_FULL'
        else:
            str_mode = 'N_GRAM'
        self.git_name = main.git_name
        # print('transaction', transaction, commit_hash, commit_th)
        if self.is_save:
            predict_result = {
                'git_name': main.git_name,
                'expected_validate_length': main.expected_validate_length,
                'most_recent': main.most_recent,
                'mode': str_mode,
                'max_epoch': main.max_epoch,
                'dim': main.dim,
                'batch_size': main.batch_size,
                'lr': main.lr,
                'is_sub_sampling': main.is_sub_sampling,
                'is_negative_sampling': main.is_negative_sampling,
                'is_check_rename': main.is_check_rename,
                'is_cosine_similarity_predict': main.is_cosine_similarity_predict,
                'is_fix': False,
                'is_use_package': main.is_use_package,
                'is_use_class_name': main.is_use_class_name,
                'is_use_return_type': main.is_use_return_type,
                'is_use_method_name': main.is_use_method_name,
                'is_use_param_type': main.is_use_param_type,
                'is_use_param_name': main.is_use_param_name,
                'version': self.version,
                "last_modified": datetime.datetime.utcnow(),
                'predict_result': pair_result,
                'commit_th': commit_th,
                'commit_hash': commit_hash,
                'is_split_train_data': main.is_split_train_data,
                'is_simple_handle_package': main.is_simple_handle_package,
                'is_simple_handle_class_name': main.is_simple_handle_class_name,
                'is_simple_handle_return_type': main.is_simple_handle_return_type,
                'is_simple_handle_method_name': main.is_simple_handle_method_name,
                'is_simple_handle_param_type': main.is_simple_handle_param_type,
                'is_simple_handle_param_name': main.is_simple_handle_param_name,
                'is_predict_with_file_level': main.is_predict_with_file_level,
                'is_mark_respective_type': main.is_mark_respective_type,
                'is_preprocessing_package': main.is_preprocessing_package,
                'is_delete_modifier': main.is_delete_modifier,
                'is_delete_void_return_type': main.is_delete_void_return_type,
                'is_casing': main.is_casing,
                'is_delete_single_subword': main.is_delete_single_subword,
                'is_delete_number_from_method_and_param': main.is_delete_number_from_method_and_param,
                'is_number_type_token_from_return_and_param_type': main.is_number_type_token_from_return_and_param_type,
                'is_delete_sub_word_number': main.is_delete_sub_word_number,
                'seed': main.seed,
                'shuffle': main.shuffle
            }
            if not self.is_load_data_from_pkl:
                self.dao.insert(predict_result)
            else:
                save_predict_result(self.git_name, self.version + '_record', commit_th)
                self.predict_result_list.append(predict_result)
                # tempの方も保存する
                save_predict_result(self.git_name, self.version + '_temp', self.predict_result_list)

    def summary(self):
        if self.is_load_data_from_pkl:
            print(self.git_name, self.version)
            save_predict_result(self.git_name, self.version, self.predict_result_list)
        return self.k_list, self.metric_list



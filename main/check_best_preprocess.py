import sys

sys.path.append('../')
import torch
import time
from model.main import Main
from model.multi_main import MultiMain
from data.mode_enum import Mode
from data.repo_data import get_repo_data
from data.data_loader import DataLoader
from data.data_store import DataStore
from data.mongo import PreprocessResultDao
import datetime


def get_param_recall(git_name, version):
    dao = PreprocessResultDao()
    result = []
    doc_list = dao.query_by(git_name, version)
    print(doc_list)
    for doc in doc_list:
        result.append(doc)
    return result


if __name__ == '__main__':
    dao = PreprocessResultDao()
    list_result = get_param_recall('tomcat', "end_version_6")
    for i in range(len(list_result)):
        param_recall = list_result[i]
        recall_list = param_recall['recall_list']
        print_str = ''
        if param_recall['is_preprocessing_package']:
            print_str += 'package预处理 '
        if param_recall['is_casing']:
            print_str += '区分大小写 '
        print(print_str, param_recall)
        for j in range(len(recall_list)):
            print(recall_list[j])

    # for i in range(len(list_result)):
    #     param_recall = list_result[i]
    #     # print('param_recall', param_recall)
    #     recall_list = param_recall['recall_list']
    #     # recall_list[0] k = 0
    #     # recall_list[1] k = 2
    #     # recall_list[2] k = 5
    #     # recall_list[3] k = 10
    #     # recall_list[4] k = 15
    #     # recall_list[5] k = 20
    #     # recall_list[6] k = 100
    #     if param_recall['is_preprocessing_package'] and \
    #        param_recall['is_delete_void_return_type'] and \
    #             (not param_recall['is_casing']) and \
    #             param_recall['is_delete_single_subword'] and \
    #             param_recall['is_delete_modifier'] and \
    #             (not param_recall['is_delete_number_from_method_and_param']) and \
    #             (not param_recall['is_number_type_token_from_return_and_param_type']):
    #         print('default', param_recall)
    #         for j in range(len(recall_list)):
    #             print(recall_list[j])
    #     if recall_list[3]['macro_recall'] > 19.5 and \
    #        recall_list[4]['macro_recall'] > 22.58 and \
    #        recall_list[5]['macro_recall'] > 24.23 and \
    #             recall_list[2]['macro_recall'] > 14.4:
    #         print_str = 'tomcat macro_recall top10 top15 top20 '
    #         # is_delete_modifier
    #         # is_delete_void_return_type
    #         # is_casing
    #         # is_delete_single_subword
    #         # is_delete_number_from_method_and_param
    #         # is_number_type_token_from_return_and_param_type
    #         if param_recall['is_preprocessing_package']:
    #             print_str += 'package预处理 '
    #         if param_recall['is_delete_void_return_type']:
    #             print_str += '删除返回值void '
    #         if param_recall['is_casing']:
    #             print_str += '区分大小写 '
    #         if param_recall['is_delete_single_subword']:
    #             print_str += '删除单个subword '
    #         if param_recall['is_delete_modifier']:
    #             print_str += '删除修饰符 '
    #         if param_recall['is_delete_number_from_method_and_param']:
    #             print_str += '从method和param里面删除数字 '
    #         if param_recall['is_number_type_token_from_return_and_param_type']:
    #             print_str += '把method和param里面的数字类型处理 '
    #         print(print_str)
    #         for j in range(len(recall_list)):
    #             print(recall_list[j])
    #     if recall_list[3]['micro_recall'] > 19 or \
    #        recall_list[4]['micro_recall'] > 22 or \
    #        recall_list[5]['micro_recall'] > 24:
    #         print('bingo')

    # list_result = get_param_recall('hive', "v1")
    # for i in range(len(list_result)):
    #     param_recall = list_result[i]
    #     # print('param_recall', param_recall)
    #     recall_list = param_recall['recall_list']
    #     # recall_list[0] k = 0
    #     # recall_list[1] k = 2
    #     # recall_list[2] k = 5
    #     # recall_list[3] k = 10
    #     # recall_list[4] k = 15
    #     # recall_list[5] k = 20
    #     # recall_list[6] k = 100
    #     if recall_list[3]['macro_recall'] > 11.74 and \
    #             recall_list[4]['macro_recall'] > 13.46 and \
    #             recall_list[5]['macro_recall'] > 14.95 and \
    #             recall_list[2]['macro_recall'] > 8:
    #         print_str = 'hive macro_recall top10 top15 top20 '
    #         # is_delete_modifier
    #         # is_delete_void_return_type
    #         # is_casing
    #         # is_delete_single_subword
    #         # is_delete_number_from_method_and_param
    #         # is_number_type_token_from_return_and_param_type
    #         if param_recall['is_preprocessing_package']:
    #             print_str += 'package预处理 '
    #         if param_recall['is_delete_void_return_type']:
    #             print_str += '删除返回值void '
    #         if param_recall['is_casing']:
    #             print_str += '区分大小写 '
    #         if param_recall['is_delete_single_subword']:
    #             print_str += '删除单个subword '
    #         if param_recall['is_delete_modifier']:
    #             print_str += '删除修饰符 '
    #         if param_recall['is_delete_number_from_method_and_param']:
    #             print_str += '从method和param里面删除数字 '
    #         if param_recall['is_number_type_token_from_return_and_param_type']:
    #             print_str += '把method和param里面的数字类型处理 '
    #         print(print_str)
    #         for j in range(len(recall_list)):
    #             print(recall_list[j])
        # if recall_list[3]['micro_recall'] > 11 and \
        #         recall_list[4]['micro_recall'] > 13 and \
        #         recall_list[5]['micro_recall'] > 14:
        #     print_str = 'hive micro_recall top10 top15 top20 '
        #     # is_delete_modifier
        #     # is_delete_void_return_type
        #     # is_casing
        #     # is_delete_single_subword
        #     # is_delete_number_from_method_and_param
        #     # is_number_type_token_from_return_and_param_type
        #     if param_recall['is_preprocessing_package']:
        #         print_str += 'package预处理 '
        #     if param_recall['is_delete_void_return_type']:
        #         print_str += '删除返回值void '
        #     if param_recall['is_casing']:
        #         print_str += '区分大小写 '
        #     if param_recall['is_delete_single_subword']:
        #         print_str += '删除单个subword '
        #     if param_recall['is_delete_modifier']:
        #         print_str += '删除修饰符 '
        #     if param_recall['is_delete_number_from_method_and_param']:
        #         print_str += '从method和param里面删除数字 '
        #     if param_recall['is_number_type_token_from_return_and_param_type']:
        #         print_str += '把method和param里面的数字类型处理 '
        #     print(print_str)
        #     for j in range(len(recall_list)):
        #         print(recall_list[j])





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
from data.mongo import ParamRecallDao
import datetime
from prettytable import PrettyTable


# def get_param_recall(git_name):
#     dao = ParamRecallDao()
#     result = []
#     doc_list = dao.query(git_name)
#     print(doc_list)
#     for doc in doc_list:
#         result.append(doc)
#     return result


def get_param_recall_by(git_name, version):
    dao = ParamRecallDao()
    result = []
    doc_list = dao.query_by(git_name, version)
    print(doc_list)
    for doc in doc_list:
        result.append(doc)
    return result


if __name__ == '__main__':
    # git_name = 'tomcat'
    git_name = 'camel'
    # 2021-12-31-tomcat-6（subword） 2022-1-2-tomcat-6（normal） 2022-1-7-tomcat-normal 2022-1-6-tomcat-subword
    # 2022-1-7-tomcat-subword-no-full 2022-1-8-tomcat-n-gram
    # 2022-1-9-hadoop-subword 2022-1-11-hadoop-normal
    # 2022-1-13-many-normal 2022-1-14-many-subword
    version = '2022-1-14-many-subword'
    list_result = get_param_recall_by(git_name, version)
    print('NORMAL')
    #     # recall_list[0] k = 1
    #     # recall_list[1] k = 2
    #     # recall_list[2] k = 5
    #     # recall_list[3] k = 10
    #     # recall_list[4] k = 15
    #     # recall_list[5] k = 20
    #     # recall_list[6] k = 100
    table = PrettyTable(
        ['固定5000コミット',
         'param',
         'preprocessing_package',
         'casing',
         'micro 1',
         'micro 5',
         'micro 10',
         'micro 15',
         'micro 20',
         'macro 1',
         'macro 5',
         'macro 10',
         'macro 15',
         'macro 20',
         ])
    print(len(list_result))
    for i in range(len(list_result)):
        param_recall = list_result[i]
        recall_list = param_recall['recall_list_new']
        k_1 = recall_list[0]
        k_2 = recall_list[1]
        k_5 = recall_list[2]
        k_10 = recall_list[3]
        k_15 = recall_list[4]
        k_20 = recall_list[5]
        k_100 = recall_list[6]
        max_epoch = param_recall['max_epoch']
        dim = param_recall['dim']
        batch_size = param_recall['batch_size']
        shuffle = param_recall['shuffle']
        is_preprocessing_package = param_recall['is_preprocessing_package']
        is_casing = param_recall['is_casing']
        if param_recall['mode'] == 'SUB_WORD':
            # print(param_recall)
            # if not (max_epoch == 10 and dim == 700 and batch_size == 256):
            #     continue
            # print(param_recall)
            # table.add_row(
            #     ['normal',
            #      (max_epoch, dim, batch_size),
            #      is_preprocessing_package,
            #      is_casing,
            #      round(k_1['micro_recall'], 2),
            #      round(k_5['micro_recall'], 2),
            #      round(k_10['micro_recall'], 2),
            #      round(k_15['micro_recall'], 2),
            #      round(k_20['micro_recall'], 2),
            #      round(k_1['macro_recall'], 2),
            #      round(k_5['macro_recall'], 2),
            #      round(k_10['macro_recall'], 2),
            #      round(k_15['macro_recall'], 2),
            #      round(k_20['macro_recall'], 2),
            #      ])
            # print(param_recall)
            if k_1['macro_recall'] >= 17 and \
                    k_5['macro_recall'] >= 30 and \
                    k_10['macro_recall'] >= 40 and \
                    k_15['macro_recall'] >= 43 \
                    and k_20['macro_recall'] >= 46:
                print(max_epoch, dim, batch_size)
                if shuffle:
                    print('数据乱序')
                else:
                    print('数据顺着来')
                if is_preprocessing_package:
                    print('预处理package')
                else:
                    print('不预处理package')
                if is_casing:
                    print('区分大小写')
                else:
                    print('不区分大小写')
                for j in range(len(recall_list)):
                    print(recall_list[j]['k'], round(recall_list[j]['micro_recall'], 2),
                          round(recall_list[j]['macro_recall'], 2))
                    # print(recall_list[j])
                print()
                table.add_row(
                    ['subword',
                     (max_epoch, dim, batch_size),
                     is_preprocessing_package,
                     is_casing,
                     round(k_1['micro_recall'], 2),
                     round(k_5['micro_recall'], 2),
                     round(k_10['micro_recall'], 2),
                     round(k_15['micro_recall'], 2),
                     round(k_20['micro_recall'], 2),
                     round(k_1['macro_recall'], 2),
                     round(k_5['macro_recall'], 2),
                     round(k_10['macro_recall'], 2),
                     round(k_15['macro_recall'], 2),
                     round(k_20['macro_recall'], 2),
                     ])
    print(table)

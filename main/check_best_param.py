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
    git_name = 'tomcat'
    version = 'tomcat_best_param_6'
    list_result = get_param_recall(git_name, version)
    # print('NORMAL')
    # for i in range(len(list_result)):
    #     param_recall = list_result[i]
    #     if param_recall['mode'] == 'NORMAL':
    #         recall_list = param_recall['recall_list']
    #         print(recall_list)
    print('other')
    # 'mode': 'NORMAL', 'max_epoch': 10, 'dim': 500, 'batch_size': 128, 'lr': 0.001,
    # 'max_epoch': 10, 'dim': 700, 'batch_size': 256, 'lr': 0.001,
    # param_recall['recall_list'][0] k=1 micro recall 8.18 | macro recall 8.73
    # param_recall['recall_list'][1] k=2 micro recall 10.88 | macro recall 11.55
    # param_recall['recall_list'][2] k=5 micro recall 15.67 | macro recall 16.19
    # param_recall['recall_list'][3] k=10 micro recall 19.87 | macro recall 19.91
    # param_recall['recall_list'][4] k=15 micro recall 22.49 | macro recall 22.58
    # param_recall['recall_list'][5] k=20 micro recall 24.27 | macro recall 24.23
    # param_recall['recall_list'][6] k=100
    # param_recall['recall_list'][5]['macro_recall'] > 24.6
    # for i in range(len(list_result)):
    #     param_recall = list_result[i]
    #     if param_recall['mode'] == 'NORMAL':
    #         recall_list = param_recall['recall_list']
    #         if param_recall['recall_list'][2]['macro_recall'] > 16.19 \
    #                 and param_recall['recall_list'][3]['macro_recall'] > 19.91 \
    #                 and param_recall['recall_list'][4]['macro_recall'] > 22.58 \
    #                 and param_recall['recall_list'][5]['macro_recall'] > 24.23 \
    #                 and param_recall['recall_list'][0]['macro_recall'] > 7.6 \
    #                 and param_recall['recall_list'][1]['macro_recall'] > 11:
    #             print(param_recall)
    #             for j in range(len(recall_list)):
    #                 print(recall_list[j])
    #  'max_epoch': 10, 'dim': 500, 'batch_size': 64, 'lr': 0.001,
    print('SUB WORD')
    #     # recall_list[0] k = 0
    #     # recall_list[1] k = 2
    #     # recall_list[2] k = 5
    #     # recall_list[3] k = 10
    #     # recall_list[4] k = 15
    #     # recall_list[5] k = 20
    #     # recall_list[6] k = 100
    for i in range(len(list_result)):
        param_recall = list_result[i]
        if param_recall['mode'] == 'SUB_WORD':
            # print(param_recall)
            recall_list = param_recall['recall_list']
            # if recall_list[5]['macro_recall'] > 25 \
            #         and recall_list[4]['macro_recall'] > 23 \
            #         and recall_list[3]['macro_recall'] > 19.5 \
            #         and i > 299:
            if recall_list[0]['macro_recall'] > 4.7 and recall_list[5]['macro_recall'] > 25 and i > 299 and recall_list[4]['macro_recall'] > 23:
                print(param_recall['max_epoch'], param_recall['dim'], param_recall['batch_size'],
                      param_recall['lr'], param_recall['last_modified'], recall_list)
                for j in range(len(recall_list)):
                    print(recall_list[j])
                # if param_recall['max_epoch'] == 10 \
                #         and param_recall['dim'] == 500 \
                #         and param_recall['batch_size'] == 64 \
                #         and param_recall['lr'] == 0.001:
                #     # print(recall_list)
                #     for j in range(len(recall_list)):
                #         print(recall_list[j])
    # print('SUB WORD NO FULL')
    # no_full_count = 0
    # for i in range(len(list_result)):
    #     param_recall = list_result[i]
    #     if param_recall['mode'] == 'SUB_WORD_NO_FULL':
    #         recall_list = param_recall['recall_list']
    #         if recall_list[2]['macro_recall'] > 12 and \
    #                 recall_list[3]['macro_recall'] > 18 \
    #                 and recall_list[4]['macro_recall'] > 22 \
    #                 and recall_list[5]['macro_recall'] > 25:
    #             print(param_recall)
    #             no_full_count += 1
    #             for j in range(len(recall_list)):
    #                 print(recall_list[j])
    # print('no_full_count', no_full_count)

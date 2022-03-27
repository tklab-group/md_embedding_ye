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
from data.mongo import SeedResultDao
import datetime
from statistics import mean, median, stdev, variance
import matplotlib.pyplot as plt


def get_param_recall(git_name, version):
    dao = SeedResultDao()
    result = []
    doc_list = dao.query_by(git_name, version)
    print(doc_list)
    for doc in doc_list:
        result.append(doc)
    return result


if __name__ == '__main__':
    dao = SeedResultDao()
    # v2, v3_clean_sub_word, v4_clean_sub_word_no_delete_number,v1_normal
    list_result = get_param_recall('tomcat', 'v1_normal')
    micro_recall_1 = []
    micro_recall_2 = []
    micro_recall_5 = []
    micro_recall_10 = []
    micro_recall_15 = []
    micro_recall_20 = []
    micro_recall_100 = []

    macro_recall_1 = []
    macro_recall_2 = []
    macro_recall_5 = []
    macro_recall_10 = []
    macro_recall_15 = []
    macro_recall_20 = []
    macro_recall_100 = []
    for i in range(len(list_result)):
        param_recall = list_result[i]
        # print('param_recall', param_recall)
        recall_list = param_recall['recall_list']
        # recall_list[0] k = 0
        # recall_list[1] k = 2
        # recall_list[2] k = 5
        # recall_list[3] k = 10
        # recall_list[4] k = 15
        # recall_list[5] k = 20
        # recall_list[6] k = 100
        micro_recall_1.append(recall_list[0]['micro_recall'])
        micro_recall_2.append(recall_list[1]['micro_recall'])
        micro_recall_5.append(recall_list[2]['micro_recall'])
        micro_recall_10.append(recall_list[3]['micro_recall'])
        micro_recall_15.append(recall_list[4]['micro_recall'])
        micro_recall_20.append(recall_list[5]['micro_recall'])
        micro_recall_100.append(recall_list[6]['micro_recall'])

        macro_recall_1.append(recall_list[0]['macro_recall'])
        macro_recall_2.append(recall_list[1]['macro_recall'])
        macro_recall_5.append(recall_list[2]['macro_recall'])
        macro_recall_10.append(recall_list[3]['macro_recall'])
        macro_recall_15.append(recall_list[4]['macro_recall'])
        macro_recall_20.append(recall_list[5]['macro_recall'])
        macro_recall_100.append(recall_list[6]['macro_recall'])

    micro_mean = {
        '1': mean(micro_recall_1),
        '2': mean(micro_recall_2),
        '5': mean(micro_recall_5),
        '10': mean(micro_recall_10),
        '15': mean(micro_recall_15),
        '20': mean(micro_recall_20),
        '100': mean(micro_recall_100),
    }

    macro_mean = {
        '1': mean(macro_recall_1),
        '2': mean(macro_recall_2),
        '5': mean(macro_recall_5),
        '10': mean(macro_recall_10),
        '15': mean(macro_recall_15),
        '20': mean(macro_recall_20),
        '100': mean(macro_recall_100),
    }

    micro_stdev = {
        '1': stdev(micro_recall_1),
        '2': stdev(micro_recall_2),
        '5': stdev(micro_recall_5),
        '10': stdev(micro_recall_10),
        '15': stdev(micro_recall_15),
        '20': stdev(micro_recall_20),
        '100': stdev(micro_recall_100),
    }

    macro_stdev = {
        '1': stdev(macro_recall_1),
        '2': stdev(macro_recall_2),
        '5': stdev(macro_recall_5),
        '10': stdev(macro_recall_10),
        '15': stdev(macro_recall_15),
        '20': stdev(macro_recall_20),
        '100': stdev(macro_recall_100),
    }

    k_list = ['1', '2', '5', '10', '15', '20', '100']
    print('mean')
    for i in range(len(k_list)):
        k = k_list[i]
        print('k', k, 'micro_recall', micro_mean[k], 'macro_recall', macro_mean[k])
    for i in range(len(k_list)):
        k = k_list[i]
        print('k', k, 'micro_recall', micro_stdev[k], 'macro_recall', macro_stdev[k])

    # 箱引け図
    fig, ax = plt.subplots()
    ax.set_title('tomcat')
    # ax.set_xticklabels(['micro recall top-1', 'macro recall top-1'])
    # ax.boxplot((micro_recall_1, macro_recall_1), showmeans=True)

    # ax.set_xticklabels(['micro recall top-5', 'macro recall top-5'])
    # ax.boxplot((micro_recall_5, macro_recall_5), showmeans=True)

    # ax.set_xticklabels(['micro recall top-10', 'macro recall top-10'])
    # ax.boxplot((micro_recall_10, macro_recall_10), showmeans=True)

    # ax.set_xticklabels(['micro recall top-15', 'macro recall top-15'])
    # ax.boxplot((micro_recall_15, macro_recall_15), showmeans=True)

    ax.set_xticklabels(['micro recall top-20', 'macro recall top-20'])
    ax.boxplot((micro_recall_20, macro_recall_20), showmeans=True)
    plt.show()
    plt.savefig







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
from data.mongo import PredictDao
import datetime
from common.util import save_data, load_data
import os
from data.util import load_predict_result
from data.git_name_version import get_git_name_version


if __name__ == '__main__':
    # git_name = 'LCExtractor'
    git_name = 'lucene'
    mode_str = 'subword'
    version_list = get_git_name_version(git_name, mode_str)
    dao = PredictDao()
    for i in range(len(version_list)):
        version = version_list[i]['version']
        print('start version', version)
        query_list = dao.query(git_name, version)
        count = 0
        for doc in query_list:
            count += 1
        print(count)
        if count == 0:
            predict_result = load_predict_result(git_name, version)
            print('load predict data len', len(predict_result))
            for j in range(len(predict_result)):
                item = predict_result[j]
                dao.insert(item)
            print('save end')
        else:
            print('data already saved')
        print('end version')

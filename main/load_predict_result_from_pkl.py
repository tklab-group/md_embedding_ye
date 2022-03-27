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
from common.util import save_data, load_data
import os
from data.util import load_predict_result, load_predict_result_from_name

if __name__ == '__main__':
    name = 'hbase2022_1_14_many_subword_hbase_6_predict.pkl'
    predict_result = load_predict_result_from_name(name)
    print(len(predict_result))
    # print(predict_result)
    print(predict_result[0])

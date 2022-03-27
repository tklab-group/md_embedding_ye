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
from data.util import load_predict_result_from_name
import os

if __name__ == '__main__':
    # pkl_name = 'tomcat2022_1_12_subword_tomcat_4_record_predict.pkl'
    # LCExtractor2021_1_14_test_LCExtractor_6_predict.pkl
    # LCExtractor2021_1_14_test_LCExtractor_6_record_predict.pkl
    # LCExtractor2021_1_14_test_LCExtractor_6_temp_predict.pkl
    # pkl_name = 'tomcat2022_1_12_normal_tomcat_9_predict.pkl'
    pkl_name = '2021_1_14_test_LCExtractor_6_predict.pkl'
    result = load_predict_result_from_name(pkl_name)
    print(len(result))
    # print(result)


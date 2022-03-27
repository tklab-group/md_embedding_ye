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
from data.mongo import MethodMapDao
from data.mongo import ModuleDataDao
from data.mongo import CoChangeDao
from data.mongo import RenameChainDao
from data.mongo import DeleteRecordDao


def save_result(data, file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    save_data(data, pkl_path)


def load_result(file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    result = load_data(pkl_path)
    return result


if __name__ == '__main__':
    git_name_list = ['hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    # git_name_list = ['tomcat']
    moduleDataDao = ModuleDataDao()
    # methodMapDao = MethodMapDao()
    # coChangeDao = CoChangeDao()
    # renameChainDao = RenameChainDao()
    # deleteRecordDao = DeleteRecordDao()
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        md_data = load_result(git_name + '_md')
        # delete_record = load_result(git_name + '_delete_record')
        # method_map = load_result(git_name + '_method_map')
        # rename_chain = load_result(git_name + '_rename_chain')
        # print(len(delete_record), len(md_data), len(method_map), len(rename_chain))
        for j in range(len(md_data)):
            # print(md_data[j])
            moduleDataDao.insert({
                'gitName': git_name,
                'list': [md_data[j]]
            })

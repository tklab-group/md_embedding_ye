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

if __name__ == '__main__':
    print(sys.path, os.getcwd())
    repo_data = get_repo_data()
    dataStore = DataStore()
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    print(project_dir)
    pkl_path_dir = project_dir + '/pkl/'
    print(pkl_path_dir)
    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        print(git_name)
        md_list = dataStore.get_module_data(git_name)
        method_map = dataStore.get_method_map(git_name)
        rename_chain_data = dataStore.get_rename_chain(git_name)
        delete_record = dataStore.get_delete_record(git_name)

        md_list_path = pkl_path_dir + git_name + '_md.pkl'
        method_map_path = pkl_path_dir + git_name + '_method_map.pkl'
        rename_chain_path = pkl_path_dir + git_name + '_rename_chain.pkl'
        delete_record_path = pkl_path_dir + git_name + '_delete_record.pkl'

        save_data(md_list, md_list_path)
        save_data(method_map, method_map_path)
        save_data(rename_chain_data, rename_chain_path)
        save_data(delete_record, delete_record_path)
        print(len(md_list), len(method_map), len(rename_chain_data), len(delete_record))
        # test load
        temp_md_list = load_data(md_list_path)
        temp_method_map = load_data(method_map_path)
        temp_rename_chain_data = load_data(rename_chain_path)
        temp_delete_record = load_data(delete_record_path)
        print(len(temp_md_list), len(temp_method_map), len(temp_rename_chain_data), len(temp_delete_record))
        # print('temp_md_list', temp_md_list)
        # print('temp_method_map', temp_method_map)
        # print('temp_rename_chain_data', temp_rename_chain_data)

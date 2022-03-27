import sys
sys.path.append('../')
from data.mongo import MethodMapDao
from data.mongo import ModuleDataDao
from data.mongo import CoChangeDao
from data.mongo import RenameChainDao
from data.mongo import DeleteRecordDao
from common.util import load_data, save_data
from config.config_default import get_config
import os


def save_predict_result(git_name, version, predict_result_list):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + git_name + version + '_predict.pkl'
    save_data(predict_result_list, pkl_path)


def load_predict_result(git_name, version):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + git_name + version + '_predict.pkl'
    result = load_data(pkl_path)
    return result


def is_predict_result_exits(git_name, version):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + git_name + version + '_predict.pkl'
    return os.path.exists(pkl_path)


def load_predict_result_from_name(pkl_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + pkl_name
    print('pkl_path', pkl_path)
    result = load_data(pkl_path)
    return result


def get_module_data(git_name):
    config = get_config()
    is_load_data_from_pkl = config['is_load_data_from_pkl']
    if not is_load_data_from_pkl:
        dao = ModuleDataDao()
        result = []
        doc_list = dao.query(git_name)
        for doc in doc_list:
            for i in range(len(doc['list'])):
                result.append(doc['list'][i])
        return result
    else:
        project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        pkl_path_dir = project_dir + '/pkl/'
        pkl_path = pkl_path_dir + git_name + '_md.pkl'
        result = load_data(pkl_path)
        print(pkl_path, len(result))
        return result


def get_method_map(git_name):
    config = get_config()
    is_load_data_from_pkl = config['is_load_data_from_pkl']
    if not is_load_data_from_pkl:
        dao = MethodMapDao()
        result = []
        doc_list = dao.query(git_name)
        for doc in doc_list:
            for i in range(len(doc['list'])):
                result.append(doc['list'][i])
        return result
    else:
        project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        pkl_path_dir = project_dir + '/pkl/'
        pkl_path = pkl_path_dir + git_name + '_method_map.pkl'
        result = load_data(pkl_path)
        print(pkl_path, len(result))
        return result


def get_rename_chain(git_name):
    config = get_config()
    is_load_data_from_pkl = config['is_load_data_from_pkl']
    if not is_load_data_from_pkl:
        dao = RenameChainDao()
        result = []
        doc_list = dao.query(git_name)
        for doc in doc_list:
            for i in range(len(doc['list'])):
                result.append(doc['list'][i])
        return result
    else:
        project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        pkl_path_dir = project_dir + '/pkl/'
        pkl_path = pkl_path_dir + git_name + '_rename_chain.pkl'
        result = load_data(pkl_path)
        print(pkl_path, len(result))
        return result


def get_delete_record(git_name):
    config = get_config()
    is_load_data_from_pkl = config['is_load_data_from_pkl']
    if not is_load_data_from_pkl:
        dao = DeleteRecordDao()
        result = []
        doc_list = dao.query(git_name)
        for doc in doc_list:
            for i in range(len(doc['list'])):
                result.append(doc['list'][i])
        return result
    else:
        project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
        pkl_path_dir = project_dir + '/pkl/'
        pkl_path = pkl_path_dir + git_name + '_delete_record.pkl'
        result = load_data(pkl_path)
        print(pkl_path, len(result))
        return result


def get_co_change(git_name):
    dao = CoChangeDao()
    result = []
    doc_list = dao.query(git_name)
    for doc in doc_list:
        for i in range(len(doc['list'])):
            result.append(doc['list'][i])
    return result


def save_module_method_map_pkl(git_name, version_name):
    module_data = get_module_data(git_name)
    method_map_data = get_method_map(git_name)
    data = {
        'module_data': module_data,
        'method_map_data': method_map_data
    }
    pkl_file_path = './model_params/pre_load_' + git_name + '_' + version_name + '.pkl'
    save_data(data, pkl_file_path)


def load_module_method_map_pkl(git_name, version_name):
    pkl_file_path = './model_params/pre_load_' + git_name + '_' + version_name + '.pkl'
    data = load_data(pkl_file_path)
    return data['module_data'], data['method_map_data']


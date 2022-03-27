import sys
sys.path.append('../')
from data.mongo import MethodMapDao
from data.mongo import ModuleDataDao
from data.mongo import CoChangeDao
from data.mongo import RenameChainDao
from common.util import load_data, save_data
from data.util import get_module_data as gmd, get_method_map as gmm, get_rename_chain as grc, get_delete_record as gdr


class DataStore:
    def __init__(self):
        self.module_data_map = {}
        self.method_map_map = {}
        self.rename_chain_map = {}
        self.delete_record_map = {}

    def get_module_data(self, git_name):
        if git_name in self.module_data_map:
            return self.module_data_map[git_name]
        else:
            md_list = gmd(git_name)
            self.module_data_map[git_name] = md_list
            return md_list

    def get_method_map(self, git_name):
        if git_name in self.method_map_map:
            return self.method_map_map[git_name]
        else:
            method_map = gmm(git_name)
            self.method_map_map[git_name] = method_map
            return method_map

    def get_rename_chain(self, git_name):
        if git_name in self.rename_chain_map:
            return self.rename_chain_map[git_name]
        else:
            rename_chain = grc(git_name)
            self.rename_chain_map[git_name] = rename_chain
            return rename_chain

    def get_delete_record(self, git_name):
        if git_name in self.delete_record_map:
            return self.delete_record_map[git_name]
        else:
            delete_record = gdr(git_name)
            self.delete_record_map[git_name] = delete_record
            return delete_record

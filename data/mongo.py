import pymongo
import sys
sys.path.append('../')
from config.config_default import get_config


class Dao:
    def __init__(self):
        config = get_config()
        dbconfig = config['database']
        connect_url = 'mongodb://{}:{}@{}:{}/?authSource={}'.format(dbconfig['user'], dbconfig['password'],
                                                                    dbconfig['host'], dbconfig['port'],
                                                                    dbconfig['database'])
        my_client = pymongo.MongoClient(connect_url)
        self.mydb = my_client[dbconfig['database']]


class MethodMapDao(Dao):
    def __init__(self):
        super(MethodMapDao, self).__init__()
        self.col = self.mydb['method_map']

    def query(self, git_name):
        my_query = {"gitName": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class ModuleDataDao(Dao):
    def __init__(self):
        super(ModuleDataDao, self).__init__()
        self.col = self.mydb['module_data']

    def query(self, git_name):
        my_query = {"gitName": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class CoChangeDao(Dao):
    def __init__(self):
        super(CoChangeDao, self).__init__()
        self.col = self.mydb['co_change_validate']

    def query(self, git_name):
        my_query = {"gitName": git_name}
        my_doc = self.col.find(my_query)
        return my_doc


class RenameChainDao(Dao):
    def __init__(self):
        super(RenameChainDao, self).__init__()
        self.col = self.mydb['rename_chain']

    def query(self, git_name):
        my_query = {"gitName": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class ParamRecallDao(Dao):
    def __init__(self):
        super(ParamRecallDao, self).__init__()
        self.col = self.mydb['param_recall']

    def query(self, git_name):
        my_query = {"git_name": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class PredictDao(Dao):
    def __init__(self):
        super(PredictDao, self).__init__()
        self.col = self.mydb['predict']

    def query(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def query_by(self, git_name, version, mode):
        my_query = {"git_name": git_name, "version": version, "mode": mode}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class PreprocessResultDao(Dao):
    def __init__(self):
        super(PreprocessResultDao, self).__init__()
        self.col = self.mydb['preprocess_result']

    def query(self, git_name):
        my_query = {"git_name": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class SeedResultDao(Dao):
    def __init__(self):
        super(SeedResultDao, self).__init__()
        self.col = self.mydb['seed_result']

    def query(self, git_name):
        my_query = {"git_name": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class YoungerResultDao(Dao):
    def __init__(self):
        super(YoungerResultDao, self).__init__()
        self.col = self.mydb['younger_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class YoungerContextsResultDao(Dao):
    def __init__(self):
        super(YoungerContextsResultDao, self).__init__()
        self.col = self.mydb['younger_contexts_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class YoungerTarmaqResultDao(Dao):
    def __init__(self):
        super(YoungerTarmaqResultDao, self).__init__()
        self.col = self.mydb['younger_tarmaq_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class YoungerTarmaqContextsResultDao(Dao):
    def __init__(self):
        super(YoungerTarmaqContextsResultDao, self).__init__()
        self.col = self.mydb['younger_tarmaq_contexts_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class LowFreqResultDao(Dao):
    def __init__(self):
        super(LowFreqResultDao, self).__init__()
        self.col = self.mydb['low_freq_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class LowFreqContextsResultDao(Dao):
    def __init__(self):
        super(LowFreqContextsResultDao, self).__init__()
        self.col = self.mydb['low_freq_contexts_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class LowFreqTarmaqResultDao(Dao):
    def __init__(self):
        super(LowFreqTarmaqResultDao, self).__init__()
        self.col = self.mydb['low_freq_tarmaq_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class LowFreqTarmaqContextsResultDao(Dao):
    def __init__(self):
        super(LowFreqTarmaqContextsResultDao, self).__init__()
        self.col = self.mydb['low_freq_tarmaq_contexts_result']

    def query_by(self, git_name, version):
        my_query = {"git_name": git_name, "version": version}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)


class DeleteRecordDao(Dao):
    def __init__(self):
        super(DeleteRecordDao, self).__init__()
        self.col = self.mydb['delete_record']

    def query(self, git_name):
        my_query = {"gitName": git_name}
        my_doc = self.col.find(my_query)
        return my_doc

    def insert(self, data):
        self.col.insert(data)



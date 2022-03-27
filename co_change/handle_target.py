import sys
sys.path.append('../')
from data.util import get_module_data, get_method_map
from data.data_divider import DataDivider
import copy
from data.util import get_co_change
from data.mongo import PredictDao
from data.id_mapped import IdMapped
from data.mode_enum import Mode


class HandleTarget:
    def __init__(self,
                 git_name,
                 expected_validate_length,
                 most_recent):
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent

        self.md_list = get_module_data(git_name)
        self.method_map = get_method_map(git_name)
        self.dataDivider = DataDivider(self.md_list, expected_validate_length, most_recent)

        self.train_data = self.dataDivider.get_train_data()
        self.validate_data = self.dataDivider.get_validate_data()

        # self.md_set = set()
        # self.build_md_set()

        self.idMapped = None
        self.build_vocab()

    def build_vocab(self):
        self.idMapped = IdMapped(self.train_data, self.method_map)

    # def build_md_set(self):
    #     self.md_set = set()
    #     for i in range(len(self.train_data)):
    #         train_data_item = self.train_data[i]
    #         for j in range(len(train_data_item)):
    #             self.md_set.add(train_data_item[j])

    # def update_md_set(self, validate_data_item):
    #     for i in range(len(validate_data_item)):
    #         self.md_set.add(validate_data_item[i])

    def is_exist(self, md_id):
        word = self.idMapped.all_id_to_word[md_id]
        # return md_id in self.md_set
        return word in self.idMapped.train_words

    def filter(self, co_change, is_fix):
        count = 0
        pop_count = 0
        result = copy.deepcopy(co_change)
        # print(co_change)
        # print(len(co_change), len(self.validate_data))
        for i in range(len(co_change)):
            commit_pre_list = co_change[i]['list']
            for j in range(len(commit_pre_list)):
                target = commit_pre_list[j]['target']
                if not self.is_exist(target):
                    # print(target, i)
                    result[i]['list'][j]['target'] = -1
                    count += 1
            if not is_fix:
                self.train_data.append(self.validate_data[i])
                if 0 < self.most_recent < len(self.train_data):
                    self.train_data.pop(0)
                    pop_count += 1
                # self.build_md_set()
                self.build_vocab()
        # print('handle target filter count', count, pop_count)
        return result

    def check_method_map(self):
        for i in range(len(self.method_map)):
            cur_i = self.method_map[i]
            for j in range(len(self.method_map)):
                cur_j = self.method_map[j]
                if cur_i['index'] != cur_j['index'] and cur_i['item'] == cur_j['item']:
                    print('cur_i', cur_i)
                    print('cur_j', cur_j)
                    print()

    # def test(self):
    #     # self.check_method_map()
    #     git_name_false = self.git_name + '_false'
    #     if self.most_recent > 0:
    #         git_name_false += '_' + str(self.most_recent)
    #     co_change = get_co_change(git_name_false)
    #
    #     commit_th = 453
    #     print(len(co_change[commit_th]['list']), co_change[commit_th])
    #     cur_expected_validate_length = self.expected_validate_length - commit_th
    #
    #     dataDivider = DataDivider(self.md_list, cur_expected_validate_length, self.most_recent)
    #     self.train_data = dataDivider.get_train_data()
    #     # self.build_md_set()
    #     self.build_vocab()
    #
    #     for i in range(len(co_change[commit_th]['list'])):
    #         item = co_change[commit_th]['list'][i]
    #         target = item['target']
    #         print(target, self.is_exist(target))
    #
    #     dao = PredictDao()
    #     query_list = dao.query_by(self.git_name, '2021_12_30_normal_tomcat_1', 'NORMAL')
    #     temp_query_list = []
    #     for doc in query_list:
    #         temp_query_list.append(doc)
    #         if doc['commit_th'] == commit_th:
    #             # print(doc['predict_result'])
    #             for j in range(len(doc['predict_result'])):
    #                 print(doc['predict_result'][j]['target'])
    #     print(len(temp_query_list), temp_query_list[commit_th])


if __name__ == '__main__':
    handleTarget = HandleTarget('tomcat', 1000, 5000)
    # handleTarget.test()
    # print(handleTarget.md_list)
    # print(handleTarget.train_data)



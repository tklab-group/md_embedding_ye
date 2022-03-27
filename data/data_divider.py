import sys
sys.path.append('../')
import numpy as np
from config.config_default import get_config
from statistics import mean, median, stdev, variance
import copy
import matplotlib.pyplot as plt


class DataDivider:
    def __init__(self, md_list, expected_validate_length, most_recent=5000):
        """
        params:
            md_list: co-change method level data with timeline from newest to oldest
            expected_validate_length: the length of validate data list's length,
                some item in md_list would no be train and validate data because its co-change
                list is too longer or just <= 1
        """
        self.md_list = md_list
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent

        self.train_data = []
        self.train_data_commit_hash_list = []
        self.validate_data = []
        self.validate_data_commit_hash_list = []

        self.train_skip = 0
        self.validate_skip = 0
        self.train_count_record = {}
        self.validate_count_record = {}
        self.all_count_record = {}

        config_all = get_config()
        self.max_co_change_num = config_all['max_co_change_num']
        self.filter_train_and_validate_data()

    def get_train_data(self):
        return self.train_data

    def get_validate_data(self):
        return self.validate_data

    def filter_train_and_validate_data(self):
        filter_count = self.get_filter_count()
        # print('filter_count', filter_count)
        # reverse index from filter_count - 1 to 0, for example, 0, 1, ..., filter_count - 1
        validate_skip = 0
        train_skip = 0
        for i in range(filter_count - 1, -1, -1):
            md_list_item = self.md_list[i]['list']
            if not self.filter_condition(md_list_item):
                # print('validate skip data', md_list_item, len(md_list_item))
                validate_skip += 1
                continue
            # print('validate_data i', i, md_list_item)
            self.validate_data.append(md_list_item)
            self.validate_data_commit_hash_list.append(self.md_list[i]['commitHash'])

        temp_train = []
        temp_train_data_commit_hash_list = []
        for i in range(len(self.md_list) - 1, -1, -1):
            if i + 1 <= filter_count:
                break
            md_list_item = self.md_list[i]['list']
            if not self.filter_condition(md_list_item):
                # print('train skip data', md_list_item, len(md_list_item))
                train_skip += 1
                continue
            # print('train_data i', i, self.md_list[i])
            temp_train.append(md_list_item)
            temp_train_data_commit_hash_list.append(self.md_list[i]['commitHash'])
        if self.most_recent == 0 or self.most_recent >= len(temp_train):
            self.train_data = temp_train
            self.train_data_commit_hash_list = temp_train_data_commit_hash_list
        else:
            train_len = len(temp_train)
            self.train_data = temp_train[train_len-self.most_recent:train_len]
            self.train_data_commit_hash_list = temp_train_data_commit_hash_list[train_len-self.most_recent:train_len]
            # print(train_len-self.most_recent, train_len)
            # print('len', len(self.train_data))
            # for i in range(len(self.train_data)):
            #     print('item', self.train_data[i])

        # print('validate_skip', validate_skip)
        # print('train_skip', train_skip)
        self.train_skip = train_skip
        self.validate_skip = validate_skip

    def get_filter_count(self):
        validate_length = 0
        cur_index = 0
        for i in range(len(self.md_list)):
            cur_index += 1
            md_list_item = self.md_list[i]['list']
            if not self.filter_condition(md_list_item):
                continue
            validate_length += 1
            if validate_length == self.expected_validate_length:
                break
        return cur_index

    def filter_condition_count(self):
        count = 0
        for i in range(len(self.md_list)):
            md_list_item = self.md_list[i]['list']
            if not self.filter_condition(md_list_item):
                continue
            count += 1
        return count

    def filter_condition(self, co_change_list):
        if len(co_change_list) <= 1 or len(co_change_list) > self.max_co_change_num:
            return False
        return True

    def count_average_num(self):
        count_list = []
        validate_count_list = []
        train_count_list = []
        count2_5 = 0
        count2_10 = 0
        count2_15 = 0
        count2_20 = 0
        count2_25 = 0
        count2_30 = 0
        for i in range(len(self.train_data)):
            train_data_item = self.train_data[i]
            len_data = len(train_data_item)
            count_list.append(len_data)
            train_count_list.append(len_data)
            if len_data <= 5:
                count2_5 += 1
            if len_data <= 10:
                count2_10 += 1
            if len_data <= 15:
                count2_15 += 1
            if len_data <= 20:
                count2_20 += 1
            if len_data <= 25:
                count2_25 += 1
            if len_data <= 30:
                count2_30 += 1
        for i in range(len(self.validate_data)):
            validate_data_item = self.validate_data[i]
            len_data = len(validate_data_item)
            count_list.append(len_data)
            validate_count_list.append(len_data)
        # データを統計
        # print('count_list', count_list)
        # print('validate_count_list', validate_count_list)
        # print('train_count_list', train_count_list)
        print('all mean %f | median %f | stdev %f | variance %f'
              % (mean(count_list), median(count_list), stdev(count_list), variance(count_list)))
        print('validate mean %f | median %f | stdev %f | variance %f'
              % (mean(validate_count_list), median(validate_count_list), stdev(validate_count_list),
                 variance(validate_count_list)))
        print('train mean %f | median %f | stdev %f | variance %f'
              % (mean(train_count_list), median(train_count_list), stdev(train_count_list), variance(train_count_list)))
        print('count2_5', count2_5)
        print('count2_10', count2_10)
        print('count2_15', count2_15)
        print('count2_20', count2_20)
        print('count2_25', count2_25)
        print('count2_30', count2_30)

    def get_using_module_data_num(self):
        result_set = set()
        for i in range(len(self.train_data)):
            train_data_item = self.train_data[i]
            for j in range(len(train_data_item)):
                result_set.add(train_data_item[j])
        for i in range(len(self.validate_data)):
            validate_data_item = self.validate_data[i]
            for j in range(len(validate_data_item)):
                result_set.add(validate_data_item[j])
        return len(result_set)

    def get_train_set_from(self, data_list):
        result_set = set()
        for i in range(len(data_list)):
            item_list = data_list[i]
            for j in range(len(item_list)):
                result_set.add(item_list[j])
        return result_set

    def count_oov_rate(self, is_fix):
        temp_train_data = copy.deepcopy(self.train_data)
        train_data_set = self.get_train_set_from(temp_train_data)
        oov_count_list = []
        for i in range(len(self.validate_data)):
            validate_data_item = self.validate_data[i]
            oov_count = 0
            for j in range(len(validate_data_item)):
                if not validate_data_item[j] in train_data_set:
                    oov_count += 1
            oov_count_list.append(oov_count)
            if not is_fix:
                temp_train_data.append(validate_data_item)
                # update train_data_set
                if self.most_recent == 0 or self.most_recent >= len(temp_train_data):
                    train_data_set = self.get_train_set_from(temp_train_data)
                else:
                    train_len = len(temp_train_data)
                    temp_train_data = temp_train_data[train_len - self.most_recent:train_len]
                    train_data_set = self.get_train_set_from(temp_train_data)
                # print(i, validate_data_item, len(temp_train_data), temp_train_data[len(temp_train_data)-1])
        print('oov', is_fix)
        print('oov mean %f | median %f | stdev %f | variance %f'
              % (mean(oov_count_list), median(oov_count_list), stdev(oov_count_list), variance(oov_count_list)))

    def to_count_record(self):
        for i in range(len(self.md_list)):
            md_list_item = self.md_list[i]['list']
            item_len = str(len(md_list_item))
            if item_len in self.all_count_record:
                self.all_count_record[item_len] += 1
            else:
                self.all_count_record[item_len] = 1
        for i in range(len(self.train_data)):
            item_len = str(len(self.train_data[i]))
            if item_len in self.train_count_record:
                self.train_count_record[item_len] += 1
            else:
                self.train_count_record[item_len] = 1
        for i in range(len(self.validate_data)):
            item_len = str(len(self.validate_data[i]))
            if item_len in self.validate_count_record:
                self.validate_count_record[item_len] += 1
            else:
                self.validate_count_record[item_len] = 1
        all_count_record_0 = 0
        all_count_record_1 = 0
        all_count_record_over_29 = 0
        total = 0
        for key in self.all_count_record.keys():
            total += self.all_count_record[key]
            if key == str(0):
                all_count_record_0 = self.all_count_record[key]
            if key == str(1):
                all_count_record_1 = self.all_count_record[key]
            if int(key) > self.max_co_change_num:
                all_count_record_over_29 += self.all_count_record[key]
        rest_count = len(self.md_list) - all_count_record_0 - all_count_record_1 - all_count_record_over_29
        print('total commit size', len(self.md_list))
        print('all_count_record_0 %d %f | all_count_record_1 %d %f | all_count_record_over_29 %d %f | rest %d %f'
              % (all_count_record_0, all_count_record_0/len(self.md_list),
                 all_count_record_1, all_count_record_1/len(self.md_list),
                 all_count_record_over_29, all_count_record_over_29/len(self.md_list),
                 rest_count, rest_count/len(self.md_list)))

    def check_can_use_commit_size(self):
        count = 0
        for i in range(len(self.md_list)):
            md_list_item = self.md_list[i]['list']
            if self.filter_condition(md_list_item):
                count += 1
        print('can use', count)

    def show_count_record_plot(self):
        check_max_co_change_num = self.max_co_change_num
        x = []
        all_count_record_over_29 = 0
        for key in self.all_count_record.keys():
            if int(key) <= check_max_co_change_num:
                x.append(int(key))
            else:
                all_count_record_over_29 += self.all_count_record[key]
        x.sort()
        y = []
        for i in range(len(x)):
            key = str(x[i])
            y.append(self.all_count_record[key])
        # 30 is over 29
        x.append(30)
        y.append(all_count_record_over_29)
        # print(x, y)
        plt.bar(x, y)
        plt.show()

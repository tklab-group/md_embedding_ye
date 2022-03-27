import sys
sys.path.append('../')
from collections import Counter
from data.mode_enum import Mode
from data.pre_process import PreProcess
from common.util import to_sub_word_list, n_gram_from_sub_word_list, \
    remove_prefix, sort_counter, decode_module_data, decode_method_signature, get_freq_package_common_part


class FreqCounter:
    def __init__(self,
                 train_data,
                 train_id_to_word,
                 preProcess: PreProcess,
                 mode=Mode.NORMAL,
                 is_predict_with_file_level=False):
        self.train_data = train_data
        self.train_id_to_word = train_id_to_word
        self.preProcess = preProcess
        self.mode = mode
        self.is_predict_with_file_level = is_predict_with_file_level
        self.common_prefix = preProcess.common_prefix

        self.word_counter = {}
        self.sub_word_counter = {}
        self.file_counter = {}

        self.word_total_count = 0
        self.sub_word_total_count = 0
        self.file_total_count = 0

        if self.is_predict_with_file_level:
            self.to_count_file()

        if mode == Mode.NORMAL:
            self.to_count_normal()
        else:
            self.to_count_normal()
            self.to_count_sub_word()

        if mode == Mode.SUB_WORD or mode == Mode.SUB_WORD_NO_FULL:
            if '' in self.sub_word_counter:
                self.sub_word_counter.pop('')
        if mode == Mode.N_GRAM:
            if '<>' in self.sub_word_counter:
                self.sub_word_counter.pop('<>')

        # self.debug_info()

    def to_count_file(self):
        self.file_total_count = len(self.preProcess.package_class_list)
        counter = Counter(self.preProcess.package_class_list)
        sort_count_list = sort_counter(counter)
        for i in range(len(sort_count_list)):
            item = sort_count_list[i]
            self.file_counter[item[0]] = item[1]

    def to_count_normal(self):
        self.sub_word_total_count = 0
        word_total_count = 0
        for i in range(len(self.train_data)):
            id_list = self.train_data[i]
            word_total_count += len(id_list)
            for j in range(len(id_list)):
                md_id = id_list[j]
                word = self.train_id_to_word[md_id]
                # count word freq
                if word in self.word_counter:
                    self.word_counter[word] += 1
                else:
                    self.word_counter[word] = 1
        self.word_total_count = word_total_count

    def to_count_sub_word(self):
        count_list = []
        sub_word_total_count = 0
        for i in range(len(self.preProcess.package_list)):
            item = self.preProcess.package_list[i]
            sub_word_list = self.preProcess.get_package_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])
        for i in range(len(self.preProcess.class_name_list)):
            item = self.preProcess.class_name_list[i]
            sub_word_list = self.preProcess.get_class_name_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])

        for i in range(len(self.preProcess.return_type_list)):
            item = self.preProcess.return_type_list[i]
            sub_word_list = self.preProcess.get_return_type_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])

        for i in range(len(self.preProcess.method_name_list)):
            item = self.preProcess.method_name_list[i]
            sub_word_list = self.preProcess.get_method_name_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])

        for i in range(len(self.preProcess.param_type_list)):
            item = self.preProcess.param_type_list[i]
            sub_word_list = self.preProcess.get_param_type_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])

        for i in range(len(self.preProcess.param_name_list)):
            item = self.preProcess.param_name_list[i]
            sub_word_list = self.preProcess.get_param_name_sub_word(item)
            sub_word_total_count += len(sub_word_list)
            for j in range(len(sub_word_list)):
                count_list.append(sub_word_list[j])

        counter = Counter(count_list)
        sort_count_list = sort_counter(counter)
        for i in range(len(sort_count_list)):
            item = sort_count_list[i]
            self.sub_word_counter[item[0]] = item[1]
        self.sub_word_total_count = sub_word_total_count




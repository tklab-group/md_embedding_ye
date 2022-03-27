import sys
sys.path.append('../')
from data.mode_enum import Mode
from common.util import get_common_prefix, get_freq_package_common_part


class IdMapped:
    def __init__(self, train_data, method_map):
        self.train_data = train_data
        self.method_map = method_map

        # all
        self.all_words = set()
        # word_to_idは信用できない
        self.all_word_to_id = {}
        # id_to_wordは信用できる
        self.all_id_to_word = {}

        # train
        self.train_words = set()
        self.train_word_to_id = {}
        self.train_id_to_word = {}

        self.build_all_word_id_map()
        self.build_id_mapped_from_train_data()

    def build_all_word_id_map(self):
        for i in range(len(self.method_map)):
            word = self.method_map[i]['item']
            word_id = self.method_map[i]['index']
            self.all_words.add(word)
            self.all_word_to_id[word] = word_id
            self.all_id_to_word[word_id] = word

    def build_id_mapped_from_train_data(self):
        for i in range(len(self.train_data)):
            id_list = self.train_data[i]
            for j in range(len(id_list)):
                md_id = id_list[j]
                word = self.all_id_to_word[md_id]
                self.train_words.add(word)
                self.train_word_to_id[word] = md_id
                self.train_id_to_word[md_id] = word

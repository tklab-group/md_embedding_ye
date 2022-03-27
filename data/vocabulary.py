import numpy as np
from config.config_default import get_config
from data.freq_counter import FreqCounter


class Vocabulary:
    def __init__(self, freqCounter: FreqCounter):
        self.freqCounter = freqCounter

        config_all = get_config()
        self.padding_id = config_all['dataset']['padding_id']
        self.padding_word = config_all['dataset']['padding_word']
        self.vocab_min = config_all['dataset']['vocab_min']
        self.sub_vocab_min = config_all['dataset']['sub_vocab_min']
        self.file_vocab_min = config_all['dataset']['file_vocab_min']

        # word => module data string, index of words is word's id
        self.words = []
        # reverse index
        self.word_to_index = {}

        # sub word => sub word or n-gram, index of sub_words is sub word's id
        self.sub_words = []
        # reverse index
        self.sub_word_to_index = {}

        # file level info of module data, including package and class name
        self.files = []
        # reverse index
        self.file_to_index = {}

        # padding
        self.words.append(self.padding_word)
        self.word_to_index[self.padding_word] = self.padding_id
        self.sub_words.append(self.padding_word)
        self.sub_word_to_index[self.padding_word] = self.padding_id
        self.files.append(self.padding_word)
        self.file_to_index[self.padding_word] = self.padding_id

        self.build_vocab_from_freq_counter()

    def add_word(self, word):
        if word in self.word_to_index:
            return
        self.word_to_index[word] = len(self.words)
        self.words.append(word)

    def add_sub_word(self, sub_word):
        if sub_word in self.sub_word_to_index:
            return
        self.sub_word_to_index[sub_word] = len(self.sub_words)
        self.sub_words.append(sub_word)

    def add_file(self, file):
        if file in self.file_to_index:
            return
        self.file_to_index[file] = len(self.files)
        self.files.append(file)

    def get_index_from_word(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        return None

    def get_index_from_sub_word(self, sub_word):
        if sub_word in self.sub_word_to_index:
            return self.sub_word_to_index[sub_word]
        return None

    def get_index_from_file(self, file):
        if file in self.file_to_index:
            return self.file_to_index[file]
        return None

    def build_vocab_from_freq_counter(self):
        word_counter_keys = list(self.freqCounter.word_counter.keys())
        for i in range(len(word_counter_keys)):
            word = word_counter_keys[i]
            if self.freqCounter.word_counter[word] >= self.vocab_min:
                self.add_word(word)

        sub_word_counter_keys = list(self.freqCounter.sub_word_counter.keys())
        for i in range(len(sub_word_counter_keys)):
            sub_word = sub_word_counter_keys[i]
            if self.freqCounter.sub_word_counter[sub_word] >= self.sub_vocab_min:
                self.add_sub_word(sub_word)

        file_counter_keys = list(self.freqCounter.file_counter.keys())
        for i in range(len(file_counter_keys)):
            file = file_counter_keys[i]
            if self.freqCounter.file_counter[file] >= self.file_vocab_min:
                self.add_file(file)

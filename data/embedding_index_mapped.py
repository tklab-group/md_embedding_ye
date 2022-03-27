import sys
sys.path.append('../')
from data.mode_enum import Mode
from data.vocabulary import Vocabulary
from data.pre_process import PreProcess
from common.util import to_sub_word, n_gram_from_sub_word_set, remove_prefix, to_n_gram_list


class EmbeddingIndexMapped:
    def __init__(self,
                 vocab: Vocabulary,
                 preProcess: PreProcess,
                 mode=Mode.NORMAL,
                 is_predict_with_file_level=False):

        self.vocab = vocab
        self.preProcess = preProcess
        self.mode = mode
        self.is_predict_with_file_level = is_predict_with_file_level

        self.in_embedding_num = 0
        self.out_embedding_num = 0

        self.in_embedding_index_to_word = {}
        self.word_to_in_embedding_index = {}

        self.out_embedding_index_to_word = {}
        self.word_to_out_embedding_index = {}

        self.word_length = len(vocab.words)
        self.sub_word_length = len(vocab.sub_words)
        self.file_length = len(vocab.files)

        # only word
        if mode == Mode.NORMAL:
            self.build_normal()
        # only sub word
        if mode == Mode.SUB_WORD_NO_FULL:
            self.build_sub_word_no_full()
        # word and sub word
        if mode == Mode.SUB_WORD or mode == Mode.N_GRAM:
            self.build_sub_word_n_gram()

    def handle_out_embedding(self):
        # handle out_embedding
        if self.is_predict_with_file_level:
            self.out_embedding_num = self.file_length
        else:
            self.out_embedding_num = self.word_length
        if self.is_predict_with_file_level:
            files = self.vocab.files
            for i in range(self.file_length):
                file = files[i]
                if file not in self.word_to_out_embedding_index:
                    self.out_embedding_index_to_word[i] = file
                    self.word_to_out_embedding_index[file] = i
        else:
            words = self.vocab.words
            for i in range(self.word_length):
                word = words[i]
                if word not in self.word_to_out_embedding_index:
                    self.out_embedding_index_to_word[i] = word
                    self.word_to_out_embedding_index[word] = i

    def build_normal(self):
        self.in_embedding_num = self.word_length
        words = self.vocab.words
        for i in range(self.word_length):
            word = words[i]
            if word not in self.word_to_in_embedding_index:
                self.in_embedding_index_to_word[i] = word
                self.word_to_in_embedding_index[word] = i
        self.handle_out_embedding()

    def build_sub_word_no_full(self):
        self.in_embedding_num = self.sub_word_length
        sub_words = self.vocab.sub_words
        for i in range(self.sub_word_length):
            sub_word = sub_words[i]
            if sub_word not in self.word_to_in_embedding_index:
                self.in_embedding_index_to_word[i] = sub_word
                self.word_to_in_embedding_index[sub_word] = i
        self.handle_out_embedding()

    def build_sub_word_n_gram(self):
        self.in_embedding_num = self.word_length + self.sub_word_length
        words = self.vocab.words
        sub_words = self.vocab.sub_words
        for i in range(self.word_length):
            word = words[i]
            if word not in self.word_to_in_embedding_index:
                self.in_embedding_index_to_word[i] = word
                self.word_to_in_embedding_index[word] = i
        for i in range(self.sub_word_length):
            sub_word = sub_words[i]
            offset_index = self.word_length + i
            if sub_word not in self.word_to_in_embedding_index:
                self.in_embedding_index_to_word[offset_index] = sub_word
                self.word_to_in_embedding_index[sub_word] = offset_index
        self.handle_out_embedding()

    def get_embedding_index_list_from_transaction(self, transaction, id_to_word):
        # only for debug
        embedding_index_list = []
        for i in range(len(transaction)):
            item = transaction[i]
            word = id_to_word[item]
            embedding_index_list.append(self.word_to_out_embedding_index[word])
        return embedding_index_list

    def get_embedding_index_list_from_target_word(self, target_word):
        return self.get_embedding_index_list_from_context_word(target_word, target_word)

    def get_embedding_index_from_context_word(self, context_word):
        if context_word in self.word_to_in_embedding_index:
            return self.word_to_in_embedding_index[context_word]
        return -1

    def get_embedding_index_list_from_context_word(self, context_word, cur_name):
        word_embedding_index_list = []
        sub_word_embedding_index_list = []

        if self.mode == Mode.NORMAL:
            if context_word in self.word_to_in_embedding_index:
                word_embedding_index_list.append(self.word_to_in_embedding_index[context_word])
        else:
            # extend context_word to sub word set
            if self.mode == Mode.N_GRAM:
                sub_word_list = to_n_gram_list(cur_name)
            else:
                sub_word_list = self.preProcess.get_module_data_sub_word(cur_name)

            if self.mode == Mode.SUB_WORD_NO_FULL:
                for i in range(len(sub_word_list)):
                    sub_word = sub_word_list[i]
                    if sub_word in self.word_to_in_embedding_index:
                        sub_word_embedding_index_list.append(self.word_to_in_embedding_index[sub_word])

            if self.mode == Mode.SUB_WORD or self.mode == Mode.N_GRAM:
                for i in range(len(sub_word_list)):
                    sub_word = sub_word_list[i]
                    if sub_word in self.word_to_in_embedding_index:
                        sub_word_embedding_index_list.append(self.word_to_in_embedding_index[sub_word])
                if context_word in self.word_to_in_embedding_index:
                    word_embedding_index_list.append(self.word_to_in_embedding_index[context_word])

        return word_embedding_index_list, sub_word_embedding_index_list

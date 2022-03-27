import numpy as np
from config.config_default import get_config
from common.util import sub_word, hump2underline, n_gram, n_gram_from_sub_word_set


# 词库第一个词语设定为<PADDING>，只是用于填充，id为0，所以要重新处理一下
class Vocabulary:
    def __init__(self, train_module_data, all_md_id_to_word, all_word_to_md_id, mode='normal'):
        self.mode = mode
        config_all = get_config()
        self.config = config_all['dataset']
        # 这里要处理罕见词，罕见词不作为词库的一部分，现在都是默认设定为1，也就是先不处理罕见词
        self.vocab_min = self.config['vocab_min']
        # <PADDING>
        self.padding_word = self.config['padding_word']
        self.padding_id = self.config['padding_id']
        # 这里的下标对应的就是word_id
        self.corpus = []
        self.corpus_length = 0
        self.word_to_id = {}
        self.id_to_word = {}
        # 这里只记录了词库的各个出现频率，数据上不完整的，要用的话需要配合md_id_to_freq和md_id_to_word_id，word_id_to_md_id来使用
        self.id_to_freq = {}
        # module data里面的id和实际上词库的id是不一样的，这里需要用map管理起来
        self.md_id_to_word_id = {}
        self.word_id_to_md_id = {}
        self.md_id_to_freq = {}
        self.md_ids = set()
        # 这个module_data，后期尽管数据加入也不进行管理了
        self.module_data = []
        # 为了之后的词库更新，这里需要保存所有的md数据
        self.all_md_id_to_word = {}
        self.all_word_to_md_id = {}
        self.build(train_module_data, all_md_id_to_word, all_word_to_md_id)
        # 在构建完没有拓展的词库之后，开始进行追加的构建，拓展出来的词都不用freq进行管理，因为都纳入模型的input embedding那里管理
        # sub word相关的，这里的sub word都是词库里面对应的
        self.sub_words = set()
        self.word_to_sub_word = {}
        self.sub_words_length = 0
        self.sub_word_to_id = {}
        self.id_to_sub_word = {}
        self.word_id_to_sub_words_id_list = {}
        # n-gram相关的，和词库里面对应的
        self.n_grams = set()
        self.word_to_n_gram = {}
        self.n_grams_length = 0
        self.n_grams_number = 3
        self.n_gram_to_id = {}
        self.id_to_n_gram = {}
        self.word_id_to_n_grams_id_list = {}
        if mode == 'sub-word' or mode == 'kobayashi':
            self.build_sub_word()
        if mode == 'n-gram':
            self.build_n_grams()

    def build_sub_word(self):
        all_sub_words = set()
        for i in range(self.corpus_length):
            sub_word_result = sub_word(hump2underline(self.corpus[i]))
            self.word_to_sub_word[self.corpus[i]] = sub_word_result
            all_sub_words = all_sub_words.union(sub_word_result)
        self.sub_words = all_sub_words
        for sub_words_item in self.sub_words:
            sub_word_id = self.corpus_length + self.sub_words_length
            self.sub_word_to_id[sub_words_item] = sub_word_id
            self.id_to_sub_word[sub_word_id] = sub_words_item
            self.sub_words_length += 1
        for i in range(self.corpus_length):
            word_id = self.word_to_id[self.corpus[i]]
            sub_words_id_list = []
            sub_word_result = self.word_to_sub_word[self.corpus[i]]
            for sub_word_result_item in sub_word_result:
                sub_words_id_list.append(self.sub_word_to_id[sub_word_result_item])
            self.word_id_to_sub_words_id_list[word_id] = sub_words_id_list

    def build_n_grams(self):
        all_n_grams = set()
        for i in range(self.corpus_length):
            n_grams_result = n_gram_from_sub_word_set(sub_word(hump2underline(self.corpus[i])))
            self.word_to_n_gram[self.corpus[i]] = n_grams_result
            all_n_grams = all_n_grams.union(n_grams_result)
        self.n_grams = all_n_grams
        for n_grams_item in self.n_grams:
            n_grams_id = self.corpus_length + self.n_grams_length
            self.n_gram_to_id[n_grams_item] = n_grams_id
            self.id_to_n_gram[n_grams_id] = n_grams_item
            self.n_grams_length += 1
        for i in range(self.corpus_length):
            word_id = self.word_to_id[self.corpus[i]]
            n_grams_id_list = []
            n_grams_result = self.word_to_n_gram[self.corpus[i]]
            for n_grams_result_item in n_grams_result:
                n_grams_id_list.append(self.n_gram_to_id[n_grams_result_item])
            self.word_id_to_n_grams_id_list[word_id] = n_grams_id_list

    def build(self, train_module_data, all_md_id_to_word, all_word_to_md_id):
        self.module_data = train_module_data
        self.all_md_id_to_word = all_md_id_to_word
        self.all_word_to_md_id = all_word_to_md_id
        self.build_md_id_set(train_module_data)
        # 根据md_id_to_freq,md_ids和all_id_to_word来构建词库corpus
        # 同时，可以顺便构建md_id_to_word_id和word_id_to_md_id
        md_id_to_freq_keys = list(self.md_id_to_freq.keys())
        # PADDING 和module data无关，就等于是把corpus里面的第一位给错开
        self.corpus.append(self.padding_word)
        self.id_to_word[self.padding_id] = self.padding_word
        self.word_to_id[self.padding_word] = self.padding_id
        self.corpus_length = 1
        for i in range(len(md_id_to_freq_keys)):
            md_id = md_id_to_freq_keys[i]
            md_freq = self.md_id_to_freq[md_id]
            if md_freq >= self.vocab_min:
                word = all_md_id_to_word[md_id]
                corpus_id = self.corpus_length
                self.corpus.append(word)
                self.md_id_to_word_id[md_id] = corpus_id
                self.word_id_to_md_id[corpus_id] = md_id
                self.id_to_word[corpus_id] = word
                self.word_to_id[word] = corpus_id
                self.id_to_freq[corpus_id] = md_freq
                self.corpus_length += 1
        return self.corpus, self.word_to_id, self.id_to_word

    def build_md_id_set(self, train_module_data):
        # 根据train_module_data的数据构建md_id_to_freq和md_ids
        # 后面的词库用md_id_to_freq来判断哪些md是可以放入的
        for i in range(len(train_module_data)):
            module_data_item = train_module_data[i]['list']
            for j in range(len(module_data_item)):
                md_id = module_data_item[j]
                if md_id in self.md_ids:
                    self.md_id_to_freq[md_id] += 1
                else:
                    self.md_ids.add(md_id)
                    self.md_id_to_freq[md_id] = 1

    # 更新词汇频率或者追加新词，返回在词库里面对应的id，用于Embedding，如果返回-1，那就代表这个词语出现次数太低，无法加入词库
    def update_word(self, word):
        word_id = -1
        if word in self.all_word_to_md_id:
            md_id = self.all_word_to_md_id[word]
            # 更新频率，这里有可能出现一种情况是，之前的train data不存在的数据在validate data里面用到了，所以会报错，因为没有记录到freq的信息
            if md_id in self.md_id_to_freq:
                self.md_id_to_freq[md_id] += 1
            else:
                self.md_id_to_freq[md_id] = 1
            # 这里有可能不在词库里面
            if md_id in self.md_id_to_word_id:
                word_id = self.md_id_to_word_id[md_id]
                # 更新频率
                self.id_to_freq[word_id] += 1
            else:
                # 如果不在词库里面，需要判断当前md数据是否可以作为corpus加入词库
                if self.md_id_to_freq[md_id] >= self.vocab_min:
                    self.corpus.append(word)
                    word_id = self.corpus_length
                    self.corpus_length += 1
                    self.id_to_freq[word_id] = self.md_id_to_freq[md_id]
                    self.word_to_id[word] = word_id
                    self.id_to_word[word_id] = word
                    # 和md那边建立连接
                    self.md_id_to_word_id[md_id] = word_id
                    self.word_id_to_md_id[word_id] = md_id
                # 如果不能加入词库的话，这时word_id是-1
        else:
            # 动态生成一个唯一的md的id，取现有的md_id的最大值+1
            max_md_id = max(self.md_ids)
            md_id = max_md_id + 1
            self.md_ids.add(md_id)
            self.all_md_id_to_word[md_id] = word
            self.all_word_to_md_id[word] = md_id
            self.md_id_to_freq[md_id] = 1
            # 根据频率来判断是否可以加入词库
            if self.md_id_to_freq[md_id] >= self.vocab_min:
                word_id = self.corpus_length
                self.corpus_length += 1
                self.md_id_to_word_id[md_id] = word_id
                self.word_id_to_md_id[word_id] = md_id
                self.corpus.append(word)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
                self.id_to_freq[word_id] = 1
        return word_id














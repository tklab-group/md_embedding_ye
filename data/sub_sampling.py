import sys
sys.path.append('../')
from data.freq_counter import FreqCounter
import random
import numpy as np
import matplotlib.pyplot as plt
from config.config_default import get_config
from common.util import sort_counter


class SubSampling:
    # Distributed Representations of Words and Phrases and their Compositionality
    # 2.3 Subsampling of Frequent Words
    # For each word we encounter in our training text,
    # there is a chance that we will effectively delete it from the text.
    # The probability that we cut the word is related to the word’s frequency.
    # Words that show up often such as “the”, “of”, and “for” don’t provide much context to the nearby words.
    # If we discard some of them, we can remove some of the noise from our data
    # and in return get faster training and better representations.
    def __init__(self,
                 freqCounter: FreqCounter,
                 sub_sampling_threshold=1e-5,
                 ):
        self.freqCounter = freqCounter
        self.threshold = sub_sampling_threshold
        print('sub sampling threshold', sub_sampling_threshold)
        self.word_drop_probability = {}
        self.sub_word_drop_probability = {}

        self.word_freq_list = {}
        self.sub_word_freq_list = {}

        self.build_drop_probability(True)
        self.build_drop_probability(False)

        config = get_config()
        self.seed = config['torch_seed']

    def build_drop_probability(self, is_sub_word=False):
        counter = self.freqCounter.word_counter
        total_count = self.freqCounter.word_total_count
        if is_sub_word:
            counter = self.freqCounter.sub_word_counter
            total_count = self.freqCounter.sub_word_total_count
        drop_probability = {}
        # for word, count in counter.items():
        #     print(word, count)
        freq_list = {word: count/total_count for word, count in counter.items()}
        # print(freq_list)
        over_count = 0
        for word in freq_list:
            if freq_list[word] < self.threshold:
                over_count += 1
                # print(word, freq_list[word])
        # print('over_count', over_count, total_count)
        drop_probability = {word: max(1 - np.sqrt(self.threshold / freq_list[word]), 0) for word in counter}
        # print('drop probability', len(drop_probability.keys()))
        # for word in drop_probability:
        #     print(word, drop_probability[word])
        # x_list = []
        # y_list = []
        # for word in drop_probability:
        #     x_list.append(freq_list[word])
        #     y_list.append(drop_probability[word])
        # plt.plot(x_list, y_list, 'ro')
        # plt.axis([0, 0.1, 0, 1])
        # plt.xlabel('frequency')
        # plt.ylabel('drop Probability')
        # plt.show()
        if not is_sub_word:
            self.word_drop_probability = drop_probability
            self.word_freq_list = freq_list
        else:
            self.sub_word_drop_probability = drop_probability
            self.sub_word_freq_list = freq_list

    def drop_probability(self, word, is_sub_word=False):
        drop_probability_set = self.word_drop_probability
        if is_sub_word:
            drop_probability_set = self.sub_word_drop_probability
        if word in drop_probability_set:
            return drop_probability_set[word]
        else:
            return 0

    def is_keep(self, word, is_sub_word=False):
        p_drop = self.drop_probability(word, is_sub_word)
        random.seed(self.seed)
        return random.random() < (1 - p_drop)




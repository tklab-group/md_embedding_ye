import sys

import torch

sys.path.append('../../')
# from data.util import get_module_data, get_method_map
import numpy as np
from config.config_default import get_config
from data.vocabulary import Vocabulary
from statistics import mean, median, stdev, variance
from data.mode_enum import Mode
from data.contexts_target_builder import ContextsTargetBuilder
from data.data_divider import DataDivider
from data.id_mapped import IdMapped
from data.freq_counter import FreqCounter
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.sub_sampling import SubSampling
import time
from tests.data.test_data2 import get_method_map, get_module_data
from data.negative_sampling import NegativeSampling
from collections import Counter


md_list = get_module_data()
method_map = get_method_map()
git_name = 'tomcat'
# md_list = get_module_data(git_name)
# method_map = get_method_map(git_name)
dataDivider = DataDivider(md_list, 3)
# print('train data', dataDivider.get_train_data())
# print('validate data', dataDivider.get_validate_data())
mode = Mode.NORMAL
is_sub_sampling = True

idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
freqCounter = FreqCounter(dataDivider.get_train_data(), idMapped.train_id_to_word, mode)
negativeSampling = NegativeSampling(freqCounter, idMapped, dataDivider.get_train_data())

negativeSampling.word_list = ['A', 'B', 'C', 'D', 'E']
origin_freq_list = [0.5, 0.2, 0.15, 0.1, 0.05]
negativeSampling.freq_list = torch.from_numpy(np.array(origin_freq_list))
freq_list = negativeSampling.freq_list.numpy()
# freq_list = negativeSampling.freq_list

print(negativeSampling.md_id_to_position_index)
print('dataDivider.get_train_data()', dataDivider.get_train_data())
query = [1]
md_id = 4
contexts = [1, 2]
target = 4
print('negative sampling', negativeSampling.sampling(contexts, target))

for i in range(len(freq_list)):
    print(negativeSampling.word_list[i], freq_list[i])
# print(negativeSampling._sampling())
negativeCounter = Counter(negativeSampling._sampling())
print(negativeCounter)





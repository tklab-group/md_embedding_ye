import sys
sys.path.append('../../')
from data.util import get_module_data, get_method_map
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
# from tests.data.test_data import get_method_map, get_module_data


# md_list = get_module_data()
# method_map = get_method_map()
git_name = 'tomcat'
md_list = get_module_data(git_name)
method_map = get_method_map(git_name)
dataDivider = DataDivider(md_list, 1000)
# print('train data', dataDivider.get_train_data())
# print('validate data', dataDivider.get_validate_data())
mode = Mode.NORMAL
is_sub_sampling = True

idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
freqCounter = FreqCounter(dataDivider.get_train_data(), idMapped.train_id_to_word, mode)
subSampling = SubSampling(freqCounter)
vocab = Vocabulary(freqCounter)
embeddingIndexMapped = EmbeddingIndexMapped(vocab, mode)
contextsTargetBuilder = ContextsTargetBuilder(
            embeddingIndexMapped, idMapped, subSampling, mode, is_sub_sampling)
contexts, target = contextsTargetBuilder.get_train_contexts_target(
            dataDivider.get_train_data())




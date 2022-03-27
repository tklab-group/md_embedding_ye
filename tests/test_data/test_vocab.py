import sys
sys.path.append('../../')
from data.data_divider import DataDivider
# from data.util import get_module_data, get_method_map
from data.id_mapped import IdMapped
from data.mode_enum import Mode
from data.freq_counter import FreqCounter
from data.pre_process import PreProcess
from data.vocabulary import Vocabulary
from tests.data.test_data5 import get_method_map, get_module_data


git_name = 'tomcat'
md_list = get_module_data()
method_map = get_method_map()
dataDivider = DataDivider(md_list, 2)
mode = Mode.SUB_WORD

idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
preProcess = PreProcess(
            dataDivider.get_train_data(),
            idMapped.train_id_to_word,
            idMapped.train_words,
            mode)
print('common_prefix', preProcess.common_prefix, preProcess.freq_package_common_part)
freqCounter = freqCounter = FreqCounter(
            train_data=dataDivider.get_train_data(),
            train_id_to_word=idMapped.train_id_to_word,
            preProcess=preProcess,
            mode=mode)
print('word_counter')
for key in iter(freqCounter.word_counter.keys()):
    print(key, freqCounter.word_counter[key])
    print(preProcess.get_module_data_sub_word(key))
print('sub_word_counter')
for key in iter(freqCounter.sub_word_counter.keys()):
    print(key, freqCounter.sub_word_counter[key])
vocabulary = Vocabulary(freqCounter)
print('word_to_index', len(vocabulary.word_to_index))
print('sub_word_to_index', len(vocabulary.sub_word_to_index))




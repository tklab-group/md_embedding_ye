import sys
sys.path.append('../../')
from data.data_divider import DataDivider
# from data.util import get_module_data, get_method_map
from data.id_mapped import IdMapped
from data.mode_enum import Mode
from data.freq_counter import FreqCounter
from tests.data.test_data import get_method_map, get_module_data


md_list = get_module_data()
method_map = get_method_map()
dataDivider = DataDivider(md_list, 4)
print('train data', dataDivider.get_train_data())
print('validate data', dataDivider.get_validate_data())
mode = Mode.NORMAL

idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
freqCounter = FreqCounter(dataDivider.get_train_data(), idMapped.train_id_to_word, mode)
print('word_total_count', freqCounter.word_total_count)
print('sub_word_total_count', freqCounter.sub_word_total_count)
print('word_counter', freqCounter.word_counter)
for key in iter(freqCounter.word_counter.keys()):
    print(idMapped.train_word_to_id[key], freqCounter.word_counter[key])
print('sub_word_counter', freqCounter.sub_word_counter)


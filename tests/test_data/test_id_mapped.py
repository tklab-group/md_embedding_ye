import sys
sys.path.append('../../')
from data.data_divider import DataDivider
from data.util import get_module_data, get_method_map
from data.id_mapped import IdMapped
from data.mode_enum import Mode
# from tests.data.test_data import get_method_map, get_module_data


md_list = get_module_data('tomcat')
method_map = get_method_map('tomcat')
dataDivider = DataDivider(md_list, 1000)
mode = Mode.SUB_WORD

# print(method_map)
# for i in range(len(method_map)):
#     print(method_map[i])

# print('train data', len(dataDivider.get_train_data()))
idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
# print('all_words', idMapped.all_words)
print('all_word_to_id', len(idMapped.all_word_to_id))
# print('all_id_to_word', idMapped.all_id_to_word)
print('train_word_to_id', len(idMapped.train_word_to_id))
# print('train_id_to_word', idMapped.train_id_to_word)
print('common_prefix', idMapped.common_prefix)

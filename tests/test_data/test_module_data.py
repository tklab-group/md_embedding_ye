import sys
sys.path.append('../../')
from data.data_divider import DataDivider
from data.util import get_module_data, get_method_map


def build_word_id_map(method_map):
    word_to_id = {}
    id_to_word = {}
    for i in range(len(method_map)):
        word = method_map[i]['item']
        word_id = method_map[i]['index']
        word_to_id[word] = word_id
        id_to_word[word_id] = word
    return word_to_id, id_to_word


git_name = 'LCExtractor'
module_data = get_module_data(git_name)
# method_map = get_method_map(git_name)
my_word_to_id, my_id_to_word = build_word_id_map(get_method_map(git_name))
# print(module_data)
# print(my_word_to_id)
# print(my_id_to_word)
# print(method_map)
print('size', len(module_data))
for i in range(len(module_data)):
    module_data_list = module_data[i]['list']
    show_array = []
    for j in range(len(module_data_list)):
        item = module_data_list[j]
        show_array.append(my_id_to_word[item])
    print(show_array)
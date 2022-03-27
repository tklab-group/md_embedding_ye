import sys
sys.path.append('../../')
from common.util import get_common_prefix_for_list

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

str1 = "src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)"
common_prefix = "src/tklab/hagward/lcextracto"
str2 = remove_prefix(str1, common_prefix)
print(str1)
print(common_prefix, str2)
if common_prefix:
    print(1)
common_prefix2 = None
if common_prefix2:
    print(2)

str3 = '333'
str4 = 'abc'
print(str3 + str4)
str_list_a = ['java', 'org', 'apache']
str_list_b = ['java', 'org']
print(get_common_prefix_for_list(str_list_a, str_list_b))
import sys
sys.path.append('../../')
from common.util import to_sub_word, to_n_gram, n_gram_from_sub_word_set, get_common_prefix, \
    to_sub_word_list, to_n_gram_list, n_gram_from_sub_word_list, hump2underline, camel_case_split

# src tklab hagward lcextractor scm mapped list public void set replaced t old item new item
word = 'src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)'
word2 = 'src/tklab/hagward/lcextractor/scm/Commit#public_Commit(RevCommit_revCommit)'
# sub_word_set = to_sub_word(word)
# print(sub_word_set)
# n_gram_set = to_n_gram('lcextractor')
# print(n_gram_set)
print(get_common_prefix(word2, word, 'm'))
# print(to_sub_word_list(word), to_sub_word(word))
# print(to_n_gram_list('lcextractor'), to_n_gram('lcextractor'))
print(n_gram_from_sub_word_list(to_sub_word_list(word)))
print(hump2underline('TESTTaXa').split('_'))
print(hump2underline('TESTTaXa'))
test_case = ['', ' ', 'lower', 'UPPER', 'Initial', 'dromedaryCase', 'CamelCase', 'ABCWordDEF', 'ABCd123Word13']
for i in range(len(test_case)):
    item = test_case[i]
    print(item, camel_case_split(item))

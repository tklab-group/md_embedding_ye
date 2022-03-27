import sys
sys.path.append('../../../')
from data.util import sub_word, hump2underline, n_gram, n_gram_from_sub_word_set

word = 'src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)'
result = hump2underline(word)
print(result)
subword_result = sub_word(result)
print('subword_result', subword_result)
n_gram_result = n_gram('embee')
n_gram_from_sub_word_set_result = n_gram_from_sub_word_set(subword_result)
# print('result', result)
# print('n_gram_result', n_gram_result)
print('n_gram_from_sub_word_set_result', n_gram_from_sub_word_set_result)
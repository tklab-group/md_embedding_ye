import sys
sys.path.append('../../')
from data.vocabulary import Vocabulary

id_to_word = {
    1: 'I',
    2: 'test',
    3: 'the',
    4: 'cbow',
    5: 'program',
    6: 'model'
}
word_to_id = {}
id_keys = list(id_to_word.keys())
for i in range(len(id_keys)):
    word_id = id_keys[i]
    word = id_to_word[word_id]
    word_to_id[word] = word_id
print(word_to_id)
module_data = [{
    # program model
    'list': [5, 6],
}, {
    # cbow program model
    'list': [4, 5, 6]
}, {
    # I cbow
    'list': [1, 4]
}, {
    # test the
    'list': [2, 3]
}, {
    # I model
    'list': [1, 6]
}, {
    # test the cbow model
    'list': [2, 3, 4, 6]
}, {
    # I test model
    'list': [1, 2, 6]
}]
validate_data_num = 2
vocab = Vocabulary(module_data[validate_data_num:], id_to_word, word_to_id)

# print('module_data[validate_data_num:]', module_data[2:])
# vocab.build(module_data[validate_data_num:], id_to_word, word_to_id)
vocab.update_word("program")

print('new word id', vocab.update_word("cbow2"))
print('corpus', vocab.corpus)
print('corpus_length', vocab.corpus_length)
print('id_to_freq', vocab.id_to_freq)
print('id_to_word', vocab.id_to_word)
print('word_to_id', vocab.word_to_id)

# 下面是原本的md相关的
print('md_ids', vocab.md_ids)
print('md_id_to_word_id', vocab.md_id_to_word_id)
print('word_id_to_md_id', vocab.word_id_to_md_id)
print('md_id_to_freq', vocab.md_id_to_freq)
print('all_md_id_to_word', vocab.all_md_id_to_word)
print('all_word_to_md_id', vocab.all_word_to_md_id)
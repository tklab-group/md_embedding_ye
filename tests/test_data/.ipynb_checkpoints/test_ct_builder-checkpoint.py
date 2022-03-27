import sys
sys.path.append('../../')
from data.data_divider import DataDivider
# from data.util import get_module_data, get_method_map
from data.id_mapped import IdMapped
from data.mode_enum import Mode
from data.freq_counter import FreqCounter
from data.vocabulary import Vocabulary
from tests.data.test_data import get_method_map, get_module_data
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.contexts_target_builder import ContextsTargetBuilder


md_list = get_module_data()
method_map = get_method_map()
dataDivider = DataDivider(md_list, 3)
mode = Mode.NORMAL

idMapped = IdMapped(dataDivider.get_train_data(), method_map, mode)
freqCounter = FreqCounter(dataDivider.get_train_data(), idMapped.train_id_to_word, mode)
# print('word_counter', freqCounter.word_counter)
# print('sub_word_counter', freqCounter.sub_word_counter)
vocabulary = Vocabulary(freqCounter)
print('word_to_index', vocabulary.word_to_index)
print('sub_word_to_index', vocabulary.sub_word_to_index)
embeddingIndexMapped = EmbeddingIndexMapped(vocabulary, mode)
# print('in_embedding_num', embeddingIndexMapped.in_embedding_num)
# print('out_embedding_num', embeddingIndexMapped.out_embedding_num)
# print('in_embedding_index_to_word', embeddingIndexMapped.in_embedding_index_to_word)
# print('word_to_in_embedding_index', embeddingIndexMapped.word_to_in_embedding_index)
# print('out_embedding_index_to_word', embeddingIndexMapped.out_embedding_index_to_word)
# print('word_to_out_embedding_index', embeddingIndexMapped.word_to_out_embedding_index)
train_data = dataDivider.get_train_data()
validate_data = dataDivider.get_validate_data()
contextsTargetBuilder = ContextsTargetBuilder(embeddingIndexMapped, idMapped, mode)
print('train data')
for i in range(len(train_data)):
    print(embeddingIndexMapped.get_embedding_index_list_from_transaction(train_data[i], idMapped.train_id_to_word))
contexts_list, target_list = contextsTargetBuilder.get_train_contexts_target(train_data)
print('contexts_list', contexts_list)
print('target_list', target_list)
print('validate data')
for i in range(len(validate_data)):
    print(embeddingIndexMapped.get_embedding_index_list_from_transaction(validate_data[i], idMapped.all_id_to_word))
result_list = contextsTargetBuilder.get_validate_contexts_target(validate_data)
print('result_list', result_list)

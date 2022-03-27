import sys
sys.path.append('../../../')
from data.vocabulary import Vocabulary
from common import config
from data.data_loader import DataLoader, create_leave_one_out_contexts_target
from common.util import create_contexts_target, to_cpu, to_gpu, save_data
from pytorch.no_extend.data_loader import WordEmbeddingDataset
from pytorch.no_extend.cbow import EmbeddingModel
import torch
import numpy
import matplotlib.pyplot as plt
import time
import numpy as np

is_can_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_can_cuda else "cpu")

id_to_word = {
    0: '<PADDING>',
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
print('md_id_to_word_id', vocab.corpus_length)
# ハイパーパラメータの設定
max_epoch = 10
# window_size = 5
hidden_size = 2
batch_size = 2
vocab_size = vocab.corpus_length
print('vocab_size', vocab_size)
train_data = []


def print_md_data(train_data_item):
    # print('train_data_item', train_data_item)
    print_md_data_result = []
    for i in range(len(train_data_item)):
        item = {
            'contexts': [vocab.word_id_to_md_id[item] for item in train_data_item[i]['contexts']],
            'target': vocab.word_id_to_md_id[train_data_item[i]['target']]
        }
        print_md_data_result.append(item)
    print('commit module data', print_md_data_result, train_data_item)


train_contexts = []
train_target = []

for i in range(len(module_data) - validate_data_num - 1, -1, -1):
    real_index = i + validate_data_num
    # print('real_index', real_index, module_data[real_index])
    result = create_leave_one_out_contexts_target(module_data[real_index]['list'], vocab.md_id_to_word_id)
    print_md_data(result)
    for j in range(len(result)):
        q = result[j]['contexts']
        # padding
        padding_array = np.arange(4 - len(q), dtype=int)
        padding_array = np.full_like(padding_array, 0)
        q = np.concatenate((q, padding_array), axis=0)
        train_contexts.append(q)
        train_target.append(result[j]['target'])

# print('train_contexts', train_contexts)
# print('train_target', train_target)

model = EmbeddingModel(vocab_size, hidden_size, device).to(device)
# print('model', model)

total_loss = 0
loss_count = 0
loss_list = []
start_time = time.time()
max_iters = len(train_contexts) // batch_size
eval_interval = 0

dataset = WordEmbeddingDataset(train_contexts, train_target)
dataloader = torch.utils.data.DataLoader(dataset, batch_size)


def train(contexts_array, target_array):
    print('train', contexts_array, target_array)
    contexts_array = torch.tensor(contexts_array)
    target_array = torch.tensor(target_array)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = model(contexts_array, target_array)
    loss = loss.requires_grad_()
    print('loss', loss)
    loss.backward()
    optimizer.step()


def print_model():
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)
    print('in_weights', model.in_embed.weight)
    print('out_weights', model.out_embed.weight)


# print_model()
train(train_contexts, train_target)
print_model()
# train(train_contexts[1:3], train_target[1:3])
# print_model()
print('md_id_to_freq', vocab.md_id_to_freq)
new_word_id = vocab.update_word('program')
vocab.update_word('program')
vocab.update_word('model')
vocab.update_word('model')
vocab.update_word('cbow')
print('md_id_to_freq', vocab.md_id_to_freq)
print('new_word_id', new_word_id)
print('vocab.md_id_to_word_id', vocab.md_id_to_word_id)

# 这里重新实例化
in_weights, out_weights = model.increment(1)
model = EmbeddingModel(vocab_size=vocab_size+1,
                       embed_size=hidden_size,
                       device=device,
                       in_weights=in_weights,
                       out_weights=out_weights).to(device)
validate_data = module_data[0:validate_data_num]
validate_contexts = []
validate_target = []
for i in range(len(validate_data)):
    result = create_leave_one_out_contexts_target(validate_data[i]['list'], vocab.md_id_to_word_id)
    for j in range(len(result)):
        q = result[j]['contexts']
        # padding
        padding_array = np.arange(4 - len(q), dtype=int)
        padding_array = np.full_like(padding_array, 0)
        q = np.concatenate((q, padding_array), axis=0)
        validate_contexts.append(q)
        validate_target.append(result[j]['target'])
print('validate_contexts', validate_contexts)
print('validate_target', validate_target)
train(validate_contexts[0:4], validate_target[0:4])
print_model()
train(validate_contexts[4:], validate_target[4:])
print_model()
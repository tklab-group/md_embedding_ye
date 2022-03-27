# coding: utf-8
import sys
sys.path.append('../../')
from common import config
from data.data_loader import DataLoader
from common.util import create_contexts_target, to_cpu, to_gpu, save_data
from pytorch.no_extend.data_loader import WordEmbeddingDataset
from pytorch.no_extend.cbow import EmbeddingModel
import torch
import numpy
import matplotlib.pyplot as plt
import time


is_can_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_can_cuda else "cpu")
print('is_can_cuda', is_can_cuda)


# ハイパーパラメータの設定
# window_size = 5
hidden_size = 100
batch_size = 32
# batch_size = 10
max_epoch = 10
git_name = 'tomcat'
validate_data_num = 5000
# git_name = 'LCExtractor'
# validate_data_num = 19

data_loader = DataLoader(git_name, validate_data_num, False)
vocab_size = data_loader.vocab.corpus_length
print('vocab_size', vocab_size)
train = data_loader.train_data
validate = data_loader.validate_data
# print('train', train)
# print('validate', validate)
contexts = train['contexts']
# print('contexts', contexts)
# for i in range(len(contexts)):
#     print('contexts', contexts[i])
target = train['target']

dataset = WordEmbeddingDataset(contexts, target)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

model = EmbeddingModel(vocab_size, hidden_size, device).to(device)
# print('model', model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print(model.parameters())
# for para in model.parameters():
#     print(para.shape)
total_loss = 0
loss_count = 0
loss_list = []
start_time = time.time()
max_iters = len(contexts) // batch_size
eval_interval = 0

for e in range(max_epoch):
    for i, (contexts, targets) in enumerate(dataloader):
        # print('contexts', contexts)
        # print('targets', targets)
        # before = model.in_embed.weight.clone().data.numpy()
        optimizer.zero_grad()
        contexts = contexts.to(device)
        targets = targets.to(device)
        loss = model(contexts, targets)
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        total_loss += loss
        loss_count += 1
        # after = model.in_embed.weight.data.numpy()
        # print('is changed', numpy.nonzero(before - after))
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad)
        # print('-------------')
        # if i % 10 == 0:
        #     print('epoch', e, 'iteration', i, loss.item())
        #     torch.save(model.state_dict(), "embedding-{}.th".format(hidden_size))
        # 評価
        if i % 10 == 0:
            eval_interval += 1
            avg_loss = total_loss / loss_count
            elapsed_time = time.time() - start_time
            print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                  % (e + 1, i + 1, max_iters, elapsed_time, avg_loss))
            loss_list.append(float(avg_loss))
            total_loss, loss_count = 0, 0


def plot(loss_list_plot, eval_interval_plot, ylim=None):
    x = numpy.arange(len(loss_list_plot))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.plot(x, loss_list_plot, label='train')
    plt.xlabel('iterations (x' + str(eval_interval_plot) + ')')
    plt.ylabel('loss')
    plt.show()


plot(loss_list, eval_interval)

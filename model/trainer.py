# coding: utf-8
import sys
sys.path.append('../')
from data.data_loader import DataLoader
from model.data_set import WordEmbeddingDataset, collate_fn
import torch
import numpy
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, model, optimizer, device, is_fix, is_print_info, shuffle=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        self.device = device
        self.is_fix = is_fix
        self.is_print_info = is_print_info
        self.shuffle = shuffle

    def fit(self,
            contexts,
            target,
            is_negative_sampling=False,
            negative_sampling=[],
            max_epoch=10,
            batch_size=32,
            eval_interval=20):
        data_size = len(contexts)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        dataset = WordEmbeddingDataset(
            contexts=contexts,
            targets=target,
            is_negative_sampling=is_negative_sampling,
            negative_sampling=negative_sampling
        )
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        # print('shuffle', self.shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=self.shuffle)
        model.train()
        start_time = time.time()
        for epoch in range(max_epoch):
            for i, (contexts, targets, negative_sampling) in enumerate(dataloader):
                # print('xxx', contexts, targets, negative_sampling)
                optimizer.zero_grad()
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                loss = None
                if is_negative_sampling:
                    negative_sampling = negative_sampling.to(self.device)
                    loss = model(contexts, targets, negative_sampling)
                    # print('loss', loss)
                else:
                    loss = model(contexts, targets)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                total_loss += loss
                loss_count += 1

                # delete caches
                del contexts, targets, negative_sampling, loss
                torch.cuda.empty_cache()
            self.current_epoch += 1
            # 評価
            # if (eval_interval is not None) and (i % eval_interval) == 0:
            avg_loss = total_loss / loss_count
            elapsed_time = time.time() - start_time
            if self.is_fix and self.is_print_info:
                print('epoch %d |  time %d[s] | loss %.2f'
                      % (self.current_epoch, elapsed_time, avg_loss))
            self.loss_list.append(float(avg_loss))
            total_loss, loss_count = 0, 0
        return total_loss, loss_count

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()

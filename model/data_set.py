import sys
sys.path.append('../')
import torch
import numpy as np


class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 contexts,
                 targets,
                 is_negative_sampling=False,
                 negative_sampling=[]
                 ):
        super(WordEmbeddingDataset, self).__init__()
        # print('contexts', contexts)
        # print('targets', targets)
        self.contexts = contexts
        self.targets = targets
        self.is_negative_sampling = is_negative_sampling
        self.negative_sampling = negative_sampling

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        # print(len(self.negative_sampling), len(self.contexts), len(self.targets))
        # print(self.is_negative_sampling, self.negative_sampling)
        # print(self.contexts[idx], self.targets[idx], self.negative_sampling[idx])
        if self.is_negative_sampling:
            return self.contexts[idx], np.append([], self.targets[idx]), self.negative_sampling[idx]
        else:
            return self.contexts[idx], self.targets[idx], []


def collate_fn(batch):
    print('batch', batch)
    batch = list(zip(batch))
    contexts = batch[0]
    target = batch[1]
    negative_sampling = batch[2]
    return contexts, target, negative_sampling

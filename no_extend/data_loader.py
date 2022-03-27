import torch
import sys
sys.path.append('..')


class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, contexts, targets):
        super(WordEmbeddingDataset, self).__init__()  # #通过父类初始化模型，然后重写两个方法
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]
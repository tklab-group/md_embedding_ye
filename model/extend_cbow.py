import sys
sys.path.append('../')
import torch
from config.config_default import get_config
import numpy
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


class ExtendEmbeddingModel(torch.nn.Module):
    def __init__(self,
                 in_embedding_num,
                 out_embedding_num,
                 dim,
                 device,
                 in_weights=None,
                 out_weights=None,
                 is_hidden_state_sum=False):
        super(ExtendEmbeddingModel, self).__init__()
        config = get_config()
        self.padding_id = config['dataset']['padding_id']
        self.padding_word = config['dataset']['padding_word']
        self.in_embedding_num = in_embedding_num
        self.out_embedding_num = out_embedding_num
        self.dim = dim
        self.device = device
        self.is_hidden_state_sum = is_hidden_state_sum

        self.seed = config['torch_seed']
        torch.manual_seed(self.seed)
        self.in_embed = torch.nn.Embedding(num_embeddings=self.in_embedding_num, embedding_dim=self.dim,
                                           padding_idx=self.padding_id, _weight=in_weights).to(self.device)
        torch.manual_seed(self.seed)
        self.out_embed = torch.nn.Linear(self.dim, self.out_embedding_num, bias=False).to(device)
        # print('out_weights is None', out_weights is None)
        if out_weights is not None:
            self.out_embed.weight.set_(out_weights)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    def forward(self, contexts, targets):
        score = self.score(contexts)
        # loss => scalar
        loss = self.loss_fn(score, targets)
        return loss

    def score(self, contexts, is_batch=True):
        # is_batch: score => batch_size*out_embedding_num
        score = self.get_hidden_state(contexts, is_batch)
        return score

    def get_hidden_state(self, contexts, is_batch=True):
        # print(self.in_embedding_num, self.dim, self.out_embedding_num)
        # contexts => batch_size * word num * sub word num
        # target = > batch_size * 1
        # print('contexts', contexts, contexts.shape)
        # idx => batch_size * word num * sub word num
        idx = contexts.long().to(self.device)
        # print('idx', idx.shape)
        # contexts_embedding.shape => [batch_size, word num, sub word num, dim]
        contexts_embedding = self.in_embed(idx)
        if is_batch:
            # print('contexts_embedding', contexts_embedding.shape)
            # contexts_embedding.shape => [batch_size, word num, dim]
            # 暫定的に平均を取る
            contexts_embedding = torch.mean(contexts_embedding, dim=2)
            # print('contexts_embedding', contexts_embedding.shape)
            # hidden_state.shape => [batch_size, dim]
            if self.is_hidden_state_sum:
                hidden_state = torch.sum(contexts_embedding, dim=1)
            else:
                hidden_state = torch.mean(contexts_embedding, dim=1)
        else:
            contexts_embedding = torch.mean(contexts_embedding, dim=1)
            if self.is_hidden_state_sum:
                hidden_state = torch.sum(contexts_embedding, dim=0)
            else:
                hidden_state = torch.mean(contexts_embedding, dim=0)
        return hidden_state

    def word_vec(self):
        return self.in_embed.weight.detach().numpy()

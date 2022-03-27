import sys

sys.path.append('../')
import torch
from config.config_default import get_config
import numpy
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


class EmbeddingModel(torch.nn.Module):
    def __init__(self,
                 in_embedding_num,
                 out_embedding_num,
                 dim,
                 device,
                 in_weights=None,
                 out_weights=None,
                 is_hidden_state_sum=False,
                 seed=6):
        super(EmbeddingModel, self).__init__()
        config = get_config()
        self.padding_id = config['dataset']['padding_id']
        self.padding_word = config['dataset']['padding_word']
        self.in_embedding_num = in_embedding_num
        self.out_embedding_num = out_embedding_num
        self.dim = dim
        self.device = device
        self.is_hidden_state_sum = is_hidden_state_sum

        self.seed = seed
        torch.manual_seed(self.seed)
        self.in_embed = torch.nn.Embedding(num_embeddings=self.in_embedding_num, embedding_dim=self.dim,
                                           padding_idx=self.padding_id, _weight=in_weights).to(self.device)
        torch.manual_seed(self.seed)
        self.out_embed = torch.nn.Embedding(num_embeddings=self.in_embedding_num, embedding_dim=self.dim,
                                            padding_idx=self.padding_id, _weight=out_weights).to(self.device)

    def forward(self, contexts, targets, negative_sampling):
        # print(contexts.shape, targets.shape, negative_sampling.shape)
        contexts_idx = contexts.long().to(self.device)
        targets_idx = targets.long().to(self.device)
        negative_sampling_idx = negative_sampling.long().to(self.device)
        # print(contexts_idx.shape, targets_idx.shape, negative_sampling_idx.shape)
        contexts_embedding = self.in_embed(contexts_idx)
        pos_embedding = self.out_embed(targets_idx)
        neg_embedding = self.out_embed(negative_sampling_idx)
        # print(contexts_embedding.shape, pos_embedding.shape, neg_embedding.shape)
        # print('pos_embedding shape', pos_embedding.shape)
        # print('neg_embedding shape', neg_embedding.shape)

        # [batch_size, dim]
        hidden_state = torch.mean(contexts_embedding, dim=1)
        # print('hidden_state.shape 1', hidden_state.shape)
        hidden_state = hidden_state.unsqueeze(2)  # [batch_size, dim, 1]
        # print('hidden_state.shape 2', hidden_state.shape)

        # [batch_size, 1, 1]
        pos_dot = torch.bmm(pos_embedding, hidden_state)
        # print('pos_dot shape', pos_dot.shape)
        # [batch_size, (window * 2)]
        # pos_dot = pos_dot.squeeze(2)

        # [batch_size, negative sampling num, 1]
        neg_dot = torch.bmm(neg_embedding, -hidden_state)
        # print('neg_dot shape', neg_dot.shape)
        # batch_size, (window * 2 * K)]
        # neg_dot = neg_dot.squeeze(2)

        # [batch_size, 1]
        log_pos = F.logsigmoid(pos_dot).sum(1)
        # print('log_pos shape', log_pos.shape)
        # [batch_size, 1]
        log_neg = F.logsigmoid(neg_dot).sum(1)
        # print('log_neg shape', log_pos.shape)
        loss = -(log_pos + log_neg)
        # [1]
        return loss.mean()

    def word_vec(self):
        return self.in_embed.weight.detach().numpy()

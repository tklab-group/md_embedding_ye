import sys

sys.path.append('../../')
import torch
from config.config_default import get_config
import numpy
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


class EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, device, in_weights=None, out_weights=None):
        super(EmbeddingModel, self).__init__()
        config = get_config()
        self.padding_id = config['dataset']['padding_id']
        self.padding_word = config['dataset']['padding_word']
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.device = device
        # 重みの初期化
        # W_in = 0.01 * numpy.random.randn(self.vocab_size, self.embed_size).astype('f')
        # W_in[0, :] = 0
        # print('W_in', W_in)
        self.in_embed = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size,
                                           padding_idx=self.padding_id, _weight=in_weights).to(self.device)
        # self.in_embed.weight.data.copy_(torch.from_numpy(W_in))
        # self.in_embed.weight.requires_grad = True
        # print(self.in_embed(torch.LongTensor([0])))
        # self.out_embed = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size,
        #                                     padding_idx=self.padding_id, _weight=out_weights).to(self.device)
        self.out_embed = torch.nn.Linear(self.embed_size, self.vocab_size, bias=False).to(device)
        print('out_weights is None', out_weights is None)
        if out_weights is None == False:
            self.out_embed.weight.set_(out_weights)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    def forward(self, contexts, targets):
        # idx = torch.LongTensor(contexts)
        # idx = torch.tensor(contexts, dtype=torch.long, device=self.device)
        idx = contexts.long().to(self.device)
        # print('idx', idx.shape)
        contexts_embedding = self.in_embed(idx)
        # print('contexts_embedding shape', contexts_embedding.shape)
        hidden_state = torch.sum(contexts_embedding, dim=1)
        # batch_size*embed_size
        # print('hidden_state', hidden_state.shape)
        # score = hidden_state.matmul(self.out_embed.weight.t())
        score = self.out_embed(hidden_state)
        # print('hidden_state', hidden_state.shape)
        # print('self.out_embed.weight.t()', self.out_embed.weight.t().shape)
        # score = F.linear(hidden_state, self.out_embed.weight, None)
        # softmax = torch.softmax(score, dim=1)
        loss = self.loss_fn(score, targets)
        # print(loss.grad_fn)  # MSELoss
        # print(loss.grad_fn.next_functions[0][0])  # Add
        # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # Expand
        # print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])  # Expand
        # print('loss requires_grad', loss.requires_grad)
        return loss

    def word_vec(self):
        return self.in_embed.weight.detach().numpy()

    def increment(self, n):
        self.vocab_size += n
        return self.increment_in_embed(n), self.increment_out_embed(n)

    def increment_in_embed(self, n):
        # factory_kwargs = {'device': self.device, 'dtype': None}
        new_param = torch.empty((n, self.embed_size)).to(self.device)
        init.normal_(new_param)
        new_weight = torch.cat([self.in_embed.weight, new_param], dim=0).to(self.device)
        return new_weight
        # with torch.no_grad():
        #     self.in_embed.weight.set_(new_weight)

    def increment_out_embed(self, n):
        # new_param = torch.empty((self.embed_size, n)).to(self.device)
        new_param = torch.empty((n, self.embed_size)).to(self.device)
        init.normal_(new_param)
        # torch.nn.Linear(self.embed_size, self.vocab_size) shape self.vocab_size x self.embed_size
        # print('self.out_embed.weight.shape', self.out_embed.weight.shape)
        # print('new_param.shape', new_param.shape)
        new_weight = torch.cat([self.out_embed.weight, new_param], dim=0).to(self.device)
        return new_weight
        # with torch.no_grad():
        #     self.out_embed.weight.set_(new_weight)

import torch
import numpy
import sys
sys.path.append('../../')

# idx = [0, 1, 2, 3, 4]
num_embeddings = 3
embedding_dim = 4
weights = numpy.arange(num_embeddings * embedding_dim)
weights = weights.reshape(num_embeddings, embedding_dim)
weights = numpy.ones_like(weights)
weights[:, 0] = 0
weights[:, 1] = 1
weights[:, 2] = 2
# weights[0, :] += 1
weights[1, :] += 1
weights[2, :] += 2
# print('weights', weights)
# in_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
in_embed = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode='mean')
in_embed.weight.data.copy_(torch.from_numpy(weights))
print('in_embed', in_embed.weight)

contexts = [
    [1, 2],
    [0]
]

hidden_state = in_embed(torch.LongTensor(contexts))
print('hidden_state', hidden_state)

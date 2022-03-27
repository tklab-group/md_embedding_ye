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
in_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
in_embed.weight.data.copy_(torch.from_numpy(weights))
print('in_embed', in_embed.weight)

out_embed = torch.nn.Embedding(embedding_dim, num_embeddings)
out_weights = numpy.arange(num_embeddings * embedding_dim)
out_weights = out_weights.reshape(embedding_dim, num_embeddings)
out_weights = numpy.ones_like(out_weights)
out_embed.weight.data.copy_(torch.from_numpy(out_weights))
print('out_embed', out_embed.weight)

batch_size = 2
window_size = 3
contexts = [
    [1, 2],
    [0]
]
print('begin')
hidden_state_array = []
for i in range(batch_size):
    hidden_state_item = in_embed(torch.LongTensor(contexts[i]))
    # hidden_state_item = torch.mean(hidden_state_item, dim=0)
    print('hidden_state_item', hidden_state_item)
    # hidden_state_array = torch.cat([hidden_state_array, hidden_state_item], dim=1)
    hidden_state_array.append(torch.mean(hidden_state_item, dim=0))
print('hidden_state_array', hidden_state_array)
# vectors = in_embed(torch.LongTensor(contexts))
# print('vectors', vectors, vectors.shape)

# hidden_state = torch.mean(vectors, dim=1)
hidden_state = torch.tensor(hidden_state_array)
print(hidden_state, hidden_state.shape)

score = hidden_state.matmul(out_embed.weight)
print('score', score)
# softmax = torch.softmax(score, dim=1)
softmax = torch.tensor(
    [
        [0.8, 0.1, 0.1],
        [0.05, 0.9, 0.05]
    ]
)
print('softmax', softmax)
targets = torch.tensor([0, 1])
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
loss = loss_fn(softmax, targets)
print('loss', loss)

word_vec = in_embed.weight.detach().numpy()
print('xx', word_vec)
idx = [-1]
print(word_vec[idx])
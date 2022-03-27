import torch
import numpy
import sys
sys.path.append('..')
num_embeddings = 3
embedding_dim = 5
x = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).double()  # input tensor
y = torch.zeros(3)  # expected output
weights = numpy.arange(num_embeddings * embedding_dim)
weights = weights.reshape(num_embeddings, embedding_dim)
weights = numpy.zeros_like(weights)
weights[1,:] = 1
print('weights', weights)
# w = torch.randn(5, 3, requires_grad=True)
w = torch.tensor(weights, requires_grad=True, dtype=float).double()
print('begin', w.grad, w)
# b = torch.randn(3, requires_grad=True)
b = torch.tensor([0.8, 0.9, 0.1], requires_grad=True).double()
z = torch.matmul(w, x)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()
print('after', w.grad)

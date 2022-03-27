import tensorflow as tf
from data.mongo import TripletDao
from data.data_loader import DataLoader
dao = TripletDao()
# result = dao.query('wicket')
result = dao.query_all()
# triplet_list = result['tripletList']
# print(result)
f = open("/Users/yejianfeng/Desktop/log.txt", 'w+')
# print(result, file=f)
print(len(result), file=f)

a = [{
    'm': 1,
    'p': 2,
    'n': 3
}, 2, 3, 4, 5]
b = [7, 8]
c = a + b
print(c)

tf.enable_eager_execution()
print('data_loader')
data_loader = DataLoader()
dataset = data_loader.get_dataset()
# for (batch, (md_list, p_list, n_list)) in enumerate(dataset):
#     # print(batch, file=f)
#     # print(triplet_list, file=f)
#     print(batch)
#     print(md_list)
#     for i in range(len(md_list)):
#         tf.print(md_list[i])
#         tf.print(p_list[i])
#         tf.print(n_list[i])
#         tf.print(1)
#     break

for (batch, batch_dataset) in enumerate(dataset):
    # print(batch, file=f)
    # print(triplet_list, file=f)
    print(batch)
    print(batch_dataset.shape())
    print(batch_dataset)
    break

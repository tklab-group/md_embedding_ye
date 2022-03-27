import numpy as np
a = [1,2,3,4,5]
b = [6,7,8]
# 这里会报错
# print(a+1)
print(a+b)
# 默认axis为0.axis在二维以上数组中才能体现出来作用.
print(np.sum(a, axis=0))
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])
print('a', a)
for i in range(4):
    print(i)       # 0
    print(a[:, i]) # [1, 9, 3]
    # h += layer.forward(contexts[:, i])
# np.sum(a, axis=0)的含义是a[0][j],a[1][j],a[2][j]对应项相加的结果
print(np.sum(a, axis=0))
# np.sum(a, axis=1)的含义是a[i][0],a[i][1],a[i][2],a[i][3]
# axis的意思是变动哪个轴
print(np.sum(a, axis=1))

dout = [1,2,3,4]
dout = np.array(dout)
# (4,)
# shape 矩阵或者数组的维数，一维数组的时候返回数组长度
print(dout.shape)
dout = dout.reshape(dout.shape[0], 1)
print(dout)

contexts = [
    [1, 2],
    [3, 4, 5],
    [6, 7, 9, 10, 11]
]

max_contents_ws = 0
for i in range(len(contexts)):
    contents_ws = len(contexts[i])
    if contents_ws > max_contents_ws:
        max_contents_ws = contents_ws

# padding
for i in range(len(contexts)):
    for j in range(max_contents_ws - len(contexts[i])):
        contexts[i].append(-1)

print('contexts', contexts)

obj = {
    1:'a',
    2:'b',
    3:'c'
}
print(len(obj))

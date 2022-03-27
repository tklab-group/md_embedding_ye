import fasttext
import numpy as np
def cosine(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


model = fasttext.load_model("../fasttext_pretrain_model/java-ftskip-dim100-ws5.bin")


def test1():
    # 100
    print(model.get_dimension())
    # as same as JDBCDao getConnect
    vector_a = model.get_word_vector('JDBCDao.getConnect')
    vector_b = model.get_word_vector('JDBCDaoGetConnect')
    vector_c = model.get_word_vector('JDBCDaogetConnect')
    # 这三者的基本都是接近1的
    print(cosine(vector_a, vector_b))
    print(cosine(vector_a, vector_c))
    print(cosine(vector_b, vector_c))


def test2():
    embedding = model.get_input_matrix()
    # 如果是不存在的词汇(JDBCDaoGetConnect)的话，这里的word_id是会变成-1的，因为这里的word_id指的是词典里面的序号
    # 如果是存在的词汇（getName）的话，subwords里面就包含了getName这个词汇在里面，所以应该是可以处理的
    word = 'getName'
    word_id = model.get_word_id(word)
    print(word_id)
    vector_word = model.get_word_vector(word)
    # 这里的word_id不是get_input_vector里面需要的index

    # vector_word_same = model.get_input_vector(word_id)
    sub_words = model.get_subwords(word)
    print(sub_words)

    vector_word = model.get_word_vector(word)
    word_index = model.get_subword_id(word)
    print(vector_word)
    print(len(embedding))
    print(model.get_input_vector(word_index))
    print(cosine(vector_word, model.get_input_vector(word_index)))
    # print(vector_word)
    # print(vector_word_same)
    # vector_all = np.zeros([100])
    # print(vector_all)
    # for sub_word in sub_words[0]:
    #     print(sub_word)
    #     # 我想拿到subword的id对应的input_matrix的index
    #     vector_all = vector_all + model.get_word_vector(sub_word)
    # print(vector_all/len(sub_words[0]))

    # vector_all_same和上面的vector_word的值基本一样
    # vector_all_same = np.mean(word_vectors[model.get_subwords(word)[1]], 0)
    # print(vector_all_same)

    # 这里只有单词和频率的信息
    # words = model.get_words()
    # print(words)

def get_NN(vector=None):
    words = model.get_words()
    print(words)


get_NN()

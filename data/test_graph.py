import matplotlib.pyplot as plt
import numpy as np


def show_plot(tarmaq, cbow, subword):
    left = np.arange(len(subword))  # numpyで横軸を設定
    labels = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']

    width = 0.3
    plt.bar(left, tarmaq, width=width, align='center', label='Association rule mining')
    plt.bar(left + width, cbow, width=width, align='center', label='CBOW model')
    plt.bar(left + 2 * width, subword, width=width, align='center', label='Subword model')
    plt.xticks(left + width, labels)
    # plt.set_ylabel('micro recall@10')
    # # plt.set_title('Scores by group and gender')
    # plt.set_xlabel('プロジェクト')
    # plt.set_xticks(x)
    # plt.set_xticklabels(labels)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # rq1 micro recall@10
    # tarmaq = [0.531, 0.473, 0.475, 0.438, 0.770, 0.605]
    # cbow = [0.539, 0.469, 0.488, 0.451, 0.798, 0.599]
    # subword = [0.505, 0.435, 0.435, 0.403, 0.783, 0.555]

    # contexts new md over 70% top-10
    # tarmaq = [0.248, 0.216, 0.288, 0.227, 0.236, 0.352]
    # cbow = [0.254, 0.238, 0.280, 0.219, 0.251, 0.362]
    # subword = [0.368, 0.292, 0.291, 0.229, 0.269, 0.360]

    # contexts new md all
    # tarmaq = []
    # cbow = []
    # subword = []

    # contexts active md over 70% top-10
    # tarmaq = [0.586, 0.526, 0.595, 0.562, 0.741, 0.794]
    # cbow = [0.624, 0.684, 0.619, 0.686, 0.900, 0.806]
    # subword = [0.631, 0.579, 0.571, 0.645, 0.868, 0.806]

    # contexts inactive md over 70% top-10
    # tarmaq = [0.675, 0.769, 0.726, 0.696, 0.948, 0.786]
    # cbow = [0.676, 0.788, 0.739, 0.698, 0.964, 0.790]
    # subword = [0.606, 0.692, 0.650, 0.625, 0.954, 0.698]

    # target inactive md over 70% top-10
    tarmaq = [0.517, 0.445, 0.448, 0.422, 0.800, 0.575]
    cbow = [0.522, 0.449, 0.465, 0.432, 0.832, 0.571]
    subword = [0.413, 0.372, 0.362, 0.331, 0.795, 0.496]
    show_plot(tarmaq, cbow, subword)

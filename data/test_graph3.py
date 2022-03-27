import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    total = [3488, 2640, 3558, 4167, 4807, 2456]
    contexts_new_over_70 = [315 / 3488, 832 / 2640, 656 / 3558, 358 / 4167, 491 / 4807, 633 / 2456]
    contexts_active_over_70 = [295 / 3488, 19 / 2640, 84 / 3558, 121 / 4167, 220 / 4807, 180 / 2456]
    contexts_inactive_over_70 = [1051 / 3488, 510 / 2640, 775 / 3558, 886 / 4167, 2273 / 4807, 649 / 2456]
    other = []
    for i in range(len(total)):
        other.append(1 - contexts_new_over_70[i] - contexts_active_over_70[i] - contexts_inactive_over_70[i])

    left = np.arange(len(total))  # numpyで横軸を設定
    labels = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']

    height = 0.2
    plt.barh(left, contexts_new_over_70, height=height, align='center', label='new over 70%')
    plt.barh(left + height, contexts_active_over_70, height=height, align='center', label='active over 70%')
    plt.barh(left + 2 * height, contexts_inactive_over_70, height=height, align='center', label='inactive over 70%')
    plt.barh(left + 3 * height, other, height=height, align='center', label='other')
    plt.yticks(left + 2 * height, labels)
    # plt.set_ylabel('micro recall@10')
    # # plt.set_title('Scores by group and gender')
    # plt.set_xlabel('プロジェクト')
    # plt.set_xticks(x)
    # plt.set_xticklabels(labels)
    plt.legend()
    plt.show()

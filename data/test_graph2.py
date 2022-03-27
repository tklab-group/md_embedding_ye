import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


if __name__ == '__main__':
    labels = ['top-1', 'top-5', 'top-10', 'top-15', 'top-20']
    # top - k & tomcat & hadoop & lucene & hbase & cassandra & camel & 平均 \\ \hline
    # 1 & 0.073 & 0.059 & 0.057 & 0.036 & 0.081 & 0.077 & 0.064 \\ \hline
    # 5 & 0.236 & 0.164 & 0.139 & 0.149 & 0.126 & 0.224 & 0.173 \\ \hline
    # 10 & 0.327 & 0.235 & 0.191 & 0.190 & 0.222 & 0.265 & 0.238 \\ \hline
    # 15 & 0.370 & 0.269 & 0.206 & 0.214 & 0.259 & 0.301 & 0.270 \\ \hline
    # 20 & 0.424 & 0.311 & 0.237 & 0.244 & 0.281 & 0.327 & 0.304 \\ \hline
    tomcat = [0.073, 0.236, 0.327, 0.370, 0.424]
    hadoop = [0.059, 0.164, 0.235, 0.269, 0.311]
    lucene = [0.057, 0.139, 0.191, 0.206, 0.237]
    hbase = [0.036, 0.149, 0.190, 0.214, 0.244]
    cassandra = [0.081, 0.126, 0.222, 0.259, 0.281]
    camel = [0.077, 0.224, 0.265, 0.301, 0.327]
    average = [0.064, 0.173, 0.238, 0.270, 0.304]
    # 设置画布大小
    # plt.figure(figsize=(20, 4))

    # tick_spacing = 4
    fig, ax = plt.subplots(1, 1)
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(tick_spacing))

    # 标题
    # plt.title("title")
    # 横坐标描述
    # plt.xlabel('top-k')
    # 纵坐标描述
    plt.ylabel('micro recall@k')

    # 这里设置线宽、线型、线条颜色、点大小等参数
    # ax.plot(x, y, label='yy', linewidth=2, color='black', marker='o', markerfacecolor='red', markersize=4)
    # ax.plot(x, z, label='zz', linewidth=2, color='red', marker='o', markerfacecolor='black', markersize=4)
    ax.plot(labels, tomcat, label='tomcat')
    ax.plot(labels, hadoop, label='hadoop')
    ax.plot(labels, lucene, label='lucene')
    ax.plot(labels, hbase, label='hbase')
    ax.plot(labels, cassandra, label='cassandra')
    ax.plot(labels, camel, label='camel')
    ax.plot(labels, average, label='average', linewidth=2, color='red', marker='o', markerfacecolor='black', markersize=4)
    # 每个数据点加标签
    # for a, b in zip(x, z):
    #     plt.text(a, b, str(b) + '%', ha='center', va='bottom', fontsize=12)
    # 只给最后一个点加标签
    # plt.text(x[-1], y[-1], y[-1], ha='center', va='bottom', fontsize=15)

    # 旋转x轴标签
    # for label in ax.get_xticklabels():
    #     label.set_rotation(30)  # 旋转30度
    #     label.set_horizontalalignment('right')  # 向右旋转

    # 图例显示及位置确定
    plt.legend(loc='upper left')
    plt.show()

import matplotlib.pyplot as plt


def save_boxplot(git_name,
                 tarmaq_recall,
                 cbow_recall_list):
    # 箱引け図
    fig, ax = plt.subplots()
    ax.set_title('test')
    label_1 = 'TARMAQ'
    label_2 = git_name
    ax.set_xticklabels([label_1, label_2])
    ax.boxplot((tarmaq_recall, cbow_recall_list),
               showmeans=True, widths=0.6)

    plt.show()


save_boxplot('test', [1, 2], [3, 4])



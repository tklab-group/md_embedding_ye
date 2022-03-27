from statistics import mean, median, stdev, variance
if __name__ == '__main__':
    # mirco_recall = 0.5 total = 100 total_hit = 40
    # 如果每个transaction里面推荐次数一样的话，那么怎么样都是和micro recall是一样的
    # 所以要测试的话，要特意设定为不平缓的
    list_1 = [
        {
            'hit': 10,
            'total': 25,
        },
        {
            'hit': 10,
            'total': 25,
        },
        {
            'hit': 20,
            'total': 50,
        },
    ]
    # micro_recall = 0.5
    recall_list = []
    for i in range(len(list_1)):
        item = list_1[i]
        hit = item['hit']
        total = item['total']
        recall = hit/total
        print(i, recall)
        recall_list.append(recall)
    print(mean(recall_list))


if __name__ == '__main__':
    total = 100
    hit = 60
    b_list = [10, 20, 30, 15, 25]
    # 平均点的情况，也就是基本是60%
    # a_1_list = [6, 12, 18, 9, 25 * 0.6]
    a_1_list = [0, 20, 5, 15, 20]
    result_list = []
    result = 0
    for i in range(len(b_list)):
        cur = a_1_list[i]/b_list[i]
        result += cur
        result_list.append(cur)
    print(result / len(b_list))

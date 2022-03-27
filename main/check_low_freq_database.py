import sys

sys.path.append('../')
from data.low_freq_trace import LowFreqTrace
from data.mongo import LowFreqTarmaqResultDao
from data.mongo import LowFreqResultDao
from data.git_name_version import get_git_name_version

def bf_value(value1, value2, value3):
    if value1 >= value2 and value1 >= value3:
        return '\\bf{' + str(value1) + '}'
    else:
        return str(value1)


def print_tex_table(tarmaq_recall, cbow_recall, subword_recall):
    print('tarmaq_recall', tarmaq_recall)
    print('cbow_recall', cbow_recall)
    print('subword_recall', subword_recall)
    k_list = [1, 5, 10, 15, 20]
    for j in range(len(k_list)):
        k = k_list[j]
        tarmaq_rate_real = round(tarmaq_recall[k], 3)
        cbow_rate_real = round(cbow_recall[k], 3)
        subword_rate_real = round(subword_recall[k], 3)
        print(
            k,
            '&',
            bf_value(tarmaq_rate_real, cbow_rate_real, subword_rate_real),
            '&',
            bf_value(cbow_rate_real, tarmaq_rate_real, subword_rate_real),
            '&',
            bf_value(subword_rate_real, cbow_rate_real, tarmaq_rate_real),
            '\\\\',
            '\hline'
        )


def main(git_name, base_tarmaq_version, base_normal_version, base_subword_version):
    expected_validate_length = 1000
    most_recent = 5000
    k_list = [1, 5, 10, 15, 20]

    lowFreqTarmaqResultDao = LowFreqTarmaqResultDao()
    lowFreqResultDao = LowFreqResultDao()

    threshold_list = [3]

    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        print('threshold', threshold)
        tarmaq_version = base_tarmaq_version + str(threshold)
        normal_version = base_normal_version + str(threshold)
        subword_version = base_subword_version + str(threshold)

        tarmaqLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
        cbowLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
        subwordLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)

        tarmaq_list = lowFreqTarmaqResultDao.query_by(git_name, tarmaq_version)
        tarmaq_count = 0
        tarmaq_commit_th_list = []
        for doc in tarmaq_list:
            tarmaq_count += 1
            tarmaqLowFreqTrace.save_target_low_freq_predict({
                'target': doc['predict_result']['target'],
                'top100_aq': doc['predict_result']['topKList']
            })
            tarmaq_commit_th_list.append(doc['commit_th'])
        tarmaq_recall = tarmaqLowFreqTrace.target_low_freq_summary(k_list)
        # print('tarmaq_count', tarmaq_count, tarmaq_commit_th_list)
        print('tarmaq_count', tarmaq_count)

        cbow_list = lowFreqResultDao.query_by(git_name, normal_version)
        cbow_count = 0
        cbow_commit_th_list = []
        for doc in cbow_list:
            cbow_count += 1
            cbow_commit_th_list.append(doc['commit_th'])
            cbowLowFreqTrace.save_target_low_freq_predict(doc['predict_result'])
        cbow_recall = cbowLowFreqTrace.target_low_freq_summary(k_list)
        # print('cbow_count', cbow_count, cbow_commit_th_list)
        print('normal_count', cbow_count)

        cbow_list = lowFreqResultDao.query_by(git_name, subword_version)
        cbow_count = 0
        cbow_commit_th_list = []
        for doc in cbow_list:
            cbow_count += 1
            cbow_commit_th_list.append(doc['commit_th'])
            subwordLowFreqTrace.save_target_low_freq_predict(doc['predict_result'])
        subword_recall = subwordLowFreqTrace.target_low_freq_summary(k_list)
        print('subword_count', cbow_count)

        # tarmaq_recall_str = ''
        # for k in tarmaq_recall:
        #     tarmaq_recall_str += str(k) + ' ' + str(round(tarmaq_recall[k] * 100, 2)) + ', '
        # print(tarmaq_recall_str)
        # # print('tarmaq_recall', tarmaq_recall)
        #
        # cbow_recall_str = ''
        # for k in cbow_recall:
        #     cbow_recall_str += str(k) + ' ' + str(round(cbow_recall[k] * 100, 2)) + ', '
        # # print('cbow_recall', cbow_recall)
        # print(cbow_recall_str)
        #
        # cbow_recall_str = ''
        # for k in subword_recall:
        #     cbow_recall_str += str(k) + ' ' + str(round(subword_recall[k] * 100, 2)) + ', '
        # # print('cbow_recall', cbow_recall)
        # print(cbow_recall_str)

        print_tex_table(tarmaq_recall, cbow_recall, subword_recall)


if __name__ == '__main__':
    # 3
    # tomcat_v2_low_freq_1_3
    # 2022_1_8_tomcat_tomcat_6_1_10_3
    # 2022_1_8_tomcat_subword_tomcat_6_1_10_3

    # 2
    # tomcat_v2_low_freq_1_2
    # 2022_1_8_tomcat_tomcat_6_1_10_2
    # 2022_1_8_tomcat_subword_tomcat_6_1_10_2

    # 1
    # tomcat_v2_low_freq_1_1
    # 2022_1_8_tomcat_tomcat_6_1_10_1
    # 2022_1_8_tomcat_subword_tomcat_6_1_10_1
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    sub_version = '_1_20_paper_'
    tarmaq_sub_version = '_1_30_fix_final_'
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        cbow_version_list = get_git_name_version(git_name, 'normal')
        subword_version_list = get_git_name_version(git_name, 'subword')
        base_tarmaq_version = git_name + tarmaq_sub_version
        base_normal_version = cbow_version_list[0]['version'] + sub_version
        base_subword_version = subword_version_list[0]['version'] + sub_version
        main(git_name, base_tarmaq_version, base_normal_version, base_subword_version)




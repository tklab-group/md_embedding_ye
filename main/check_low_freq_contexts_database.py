import sys

sys.path.append('../')
from data.low_freq_trace import LowFreqTrace
from data.mongo import LowFreqTarmaqContextsResultDao
from data.mongo import LowFreqContextsResultDao
from data.git_name_version import get_git_name_version
from statistics import mean, median, stdev, variance


def print_recall(recall, tag):
    recall_str = tag + ''
    for k in recall:
        hit, total, rate = recall[k]
        recall_str += str(k) + ', (' + str(hit) + ', ' + str(total) + ', ' + str(round(rate * 100, 2)) + '), '
    print(recall_str)


def bf_value(value1, value2, value3):
    if value1 >= value2 and value1 >= value3:
        return '\\bf{' + str(value1) + '}'
    else:
        return str(value1)


def print_tex_table(tarmaq_recall, cbow_recall, subword_recall):
    print('tarmaq_recall', tarmaq_recall)
    print('cbow_recall', cbow_recall)
    print('subword_recall', subword_recall)
    # new old low_freq
    tag_list = ['new', 'old', 'low_freq']
    k_list = [1, 5, 10, 15, 20]
    for i in range(len(tag_list)):
        tag = tag_list[i]
        print('tag', tag)
        for j in range(len(k_list)):
            k = k_list[j]
            tarmaq_hit, tarmaq_total, tarmaq_rate = tarmaq_recall[tag][k]
            tarmaq_rate_real = round(tarmaq_hit/tarmaq_total, 3)
            cbow_hit, cbow_total, cbow_rate = cbow_recall[tag][k]
            cbow_rate_real = round(cbow_hit/cbow_total, 3)
            subword_hit, subword_total, subword_rate = subword_recall[tag][k]
            subword_rate_real = round(subword_hit/subword_total, 3)
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


def main(git_name, base_tarmaq_version, base_normal_version, base_subword_version, over_threshold_list):
    print('main', git_name, base_tarmaq_version, base_normal_version, base_subword_version)
    expected_validate_length = 1000
    most_recent = 5000
    k_list = [1, 5, 10, 15, 20]
    threshold_list = [3]

    lowFreqTarmaqContextsResultDao = LowFreqTarmaqContextsResultDao()
    lowFreqContextsResultDao = LowFreqContextsResultDao()

    result = {}
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        print('threshold=', threshold)
        for j in range(len(over_threshold_list)):
            over_threshold = over_threshold_list[j]
            print('over_threshold=', over_threshold)
            tarmaqLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
            cbowLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
            subwordLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)

            tarmaq_version = base_tarmaq_version + str(threshold)
            tarmaq_list = lowFreqTarmaqContextsResultDao.query_by(git_name, tarmaq_version)
            tarmaq_count = 0
            tarmaq_commit_th_target_minus_one = {}
            for doc in tarmaq_list:
                tarmaq_count += 1
                commit_th = doc['commit_th']
                cur_expected_validate_length = doc['cur_expected_validate_length']
                # if commit_th == 453 or commit_th == 764:
                #     print('tarmaq', commit_th, cur_expected_validate_length, doc['predict_result'])
                if doc['predict_result']['target'] == -1:
                    if commit_th in tarmaq_commit_th_target_minus_one:
                        tarmaq_commit_th_target_minus_one[commit_th] += 1
                    else:
                        tarmaq_commit_th_target_minus_one[commit_th] = 1
                tarmaqLowFreqTrace.save_contexts_component_predict({
                    'contexts_component': {
                        'new_rate': doc['new_rate'],
                        'new_list': doc['new_list'],
                        'old_rate': doc['old_rate'],
                        'old_list': doc['old_list'],
                        'low_freq_rate': doc['low_freq_rate'],
                        'low_freq_list': doc['low_freq_list'],
                    },
                    'predict_result': {
                        'target': doc['predict_result']['target'],
                        'top100_aq': doc['predict_result']['topKList']
                    },
                    'commit_th': doc['commit_th']
                })
            # print('tarmaq contexts_component_summary', tarmaq_count)
            tarmaq_recall = tarmaqLowFreqTrace.contexts_component_summary(k_list, over_threshold=over_threshold)
            # print('tarmaq_count', tarmaq_count)
            # print()
            # print('tarmaq')
            # print_recall(tarmaq_recall['new'], 'new ')
            # print_recall(tarmaq_recall['old'], 'old ')
            # print_recall(tarmaq_recall['low_freq'], 'low_freq ')
            # print()

            cbow_version = base_normal_version + str(threshold)
            cbow_list = lowFreqContextsResultDao.query_by(git_name, cbow_version)
            cbow_count = 0
            cbow_commit_th_target_minus_one = {}
            for doc in cbow_list:
                cbow_count += 1
                commit_th = doc['commit_th']
                cur_expected_validate_length = doc['cur_expected_validate_length']
                # if commit_th == 453 or commit_th == 764:
                #     print('cbow', commit_th, cur_expected_validate_length, doc['predict_result'])
                if doc['predict_result']['target'] == -1:
                    if commit_th in cbow_commit_th_target_minus_one:
                        cbow_commit_th_target_minus_one[commit_th] += 1
                    else:
                        cbow_commit_th_target_minus_one[commit_th] = 1
                cbowLowFreqTrace.save_contexts_component_predict({
                    'contexts_component': {
                        'new_rate': doc['new_rate'],
                        'new_list': doc['new_list'],
                        'old_rate': doc['old_rate'],
                        'old_list': doc['old_list'],
                        'low_freq_rate': doc['low_freq_rate'],
                        'low_freq_list': doc['low_freq_list'],
                    },
                    'predict_result': doc['predict_result'],
                    'commit_th': doc['commit_th']
                })
            # test
            # print('tarmaq_commit_th_target_minus_one', len(tarmaq_commit_th_target_minus_one))
            # for commit_th in tarmaq_commit_th_target_minus_one:
            #     print(commit_th, tarmaq_commit_th_target_minus_one[commit_th])
            # print('cbow_commit_th_target_minus_one', len(cbow_commit_th_target_minus_one))
            # for commit_th in cbow_commit_th_target_minus_one:
            #     print(commit_th, cbow_commit_th_target_minus_one[commit_th])
            # for commit_th in tarmaq_commit_th_target_minus_one:
            #     if tarmaq_commit_th_target_minus_one[commit_th] != cbow_commit_th_target_minus_one[commit_th]:
            #         print(commit_th, tarmaq_commit_th_target_minus_one[commit_th],
            #               cbow_commit_th_target_minus_one[commit_th])

            # print('cbow contexts_component_summary', cbow_count)
            cbow_recall = cbowLowFreqTrace.contexts_component_summary(k_list, over_threshold=over_threshold)
            # print('cbow_count', cbow_count)
            # print()

            # print('normal')
            # print_recall(cbow_recall['new'], 'new ')
            # print_recall(cbow_recall['old'], 'old ')
            # print_recall(cbow_recall['low_freq'], 'low_freq ')
            # print()

            cbow_version = base_subword_version + str(threshold)
            cbow_list = lowFreqContextsResultDao.query_by(git_name, cbow_version)
            cbow_count = 0
            cbow_commit_th_target_minus_one = {}
            for doc in cbow_list:
                cbow_count += 1
                commit_th = doc['commit_th']
                cur_expected_validate_length = doc['cur_expected_validate_length']
                if doc['predict_result']['target'] == -1:
                    if commit_th in cbow_commit_th_target_minus_one:
                        cbow_commit_th_target_minus_one[commit_th] += 1
                    else:
                        cbow_commit_th_target_minus_one[commit_th] = 1
                subwordLowFreqTrace.save_contexts_component_predict({
                    'contexts_component': {
                        'new_rate': doc['new_rate'],
                        'new_list': doc['new_list'],
                        'old_rate': doc['old_rate'],
                        'old_list': doc['old_list'],
                        'low_freq_rate': doc['low_freq_rate'],
                        'low_freq_list': doc['low_freq_list'],
                    },
                    'predict_result': doc['predict_result'],
                    'commit_th': doc['commit_th']
                })
            subword_recall = subwordLowFreqTrace.contexts_component_summary(k_list, over_threshold=over_threshold)
            # print('subword')
            # print_recall(subword_recall['new'], 'new ')
            # print_recall(subword_recall['old'], 'old ')
            # print_recall(subword_recall['low_freq'], 'low_freq ')
            # print()
            print_tex_table(tarmaq_recall, cbow_recall, subword_recall)
            result[over_threshold] = {
                'tarmaq': tarmaq_recall,
                'cbow': cbow_recall,
                'subword': subword_recall
            }
    return result


def analyse(git_map, tag):
    tarmaq_list_map = {}
    cbow_list_map = {}
    subword_list_map = {}

    tarmaq_result_map = {}
    cbow_result_map = {}
    subword_result_map = {}
    for git_name in git_map:
        map_result = git_map[git_name]
        for over_threshold in map_result:
            tarmaq_hit, tarmaq_total, tarmaq_rate = map_result[over_threshold]['tarmaq'][tag][10]
            cbow_hit, cbow_total, cbow_rate = map_result[over_threshold]['cbow'][tag][10]
            subword_hit, subword_total, subword_rate = map_result[over_threshold]['subword'][tag][10]
            tarmaq_rate_real = round(tarmaq_hit/tarmaq_total, 3)
            cbow_rate_real = round(cbow_hit/cbow_total, 3)
            subword_rate_real = round(subword_hit/subword_total, 3)
            if over_threshold in tarmaq_list_map:
                tarmaq_list_map[over_threshold].append(tarmaq_rate_real)
            else:
                tarmaq_list_map[over_threshold] = [tarmaq_rate_real]
            if over_threshold in cbow_list_map:
                cbow_list_map[over_threshold].append(cbow_rate_real)
            else:
                cbow_list_map[over_threshold] = [cbow_rate_real]
            if over_threshold in subword_list_map:
                subword_list_map[over_threshold].append(subword_rate_real)
            else:
                subword_list_map[over_threshold] = [subword_rate_real]
    for over_threshold in tarmaq_list_map:
        tarmaq_result_map[over_threshold] = mean(tarmaq_list_map[over_threshold])
    for over_threshold in cbow_list_map:
        cbow_result_map[over_threshold] = mean(cbow_list_map[over_threshold])
    for over_threshold in subword_list_map:
        subword_result_map[over_threshold] = mean(subword_list_map[over_threshold])
    return tarmaq_result_map, cbow_result_map, subword_result_map


if __name__ == '__main__':
    # tarmaq_version = 'tomcat_v1_low_freq_1'
    # cbow_version = '2022_1_8_tomcat_tomcat_6_low_freq_2022_1_9'
    # # cbow_version = '2022_1_8_tomcat_subword_tomcat_6_low_freq_2022_1_9'

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
    # git_name_list = ['tomcat']
    sub_version = '_1_20_paper_'
    tarmaq_sub_version = '_1_30_fix_final_'
    git_name_result_map = {}
    # over_threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # over_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    over_threshold_list = [0.7]
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        cbow_version_list = get_git_name_version(git_name, 'normal')
        subword_version_list = get_git_name_version(git_name, 'subword')
        base_tarmaq_version = git_name + tarmaq_sub_version
        base_normal_version = cbow_version_list[0]['version'] + sub_version
        base_subword_version = subword_version_list[0]['version'] + sub_version
        result = main(git_name, base_tarmaq_version, base_normal_version, base_subword_version, over_threshold_list)
        git_name_result_map[git_name] = result
    print('git_name_result_map', git_name_result_map)
    # tag_list = ['new', 'old', 'low_freq']

    tag_list = ['new', 'old', 'low_freq']
    for i in range(len(tag_list)):
        tag = tag_list[i]
        tarmaq_tag_result, cbow_tag_result, subword_tag_result = analyse(git_name_result_map, tag)
        tarmaq_result = []
        cbow_result = []
        subword_result = []
        for j in range(len(over_threshold_list)):
            over_threshold = over_threshold_list[j]
            tarmaq_result.append(tarmaq_tag_result[over_threshold])
            cbow_result.append(cbow_tag_result[over_threshold])
            subword_result.append(subword_tag_result[over_threshold])
        print('tag', tag)
        print('tarmaq_result', tarmaq_result)
        print('cbow_result', cbow_result)
        print('subword_result', subword_result)





import sys

sys.path.append('../')
from data.low_freq_trace import LowFreqTrace
from data.mongo import LowFreqTarmaqContextsResultDao
from data.mongo import LowFreqContextsResultDao
from data.git_name_version import get_git_name_version


def main(git_name, base_tarmaq_version):
    print('main', git_name, base_tarmaq_version)
    expected_validate_length = 1000
    most_recent = 5000
    k_list = [1, 5, 10, 15, 20]
    over_threshold_list = [0.7]
    threshold_list = [3]

    lowFreqTarmaqContextsResultDao = LowFreqTarmaqContextsResultDao()

    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        print('threshold=', threshold)
        for j in range(len(over_threshold_list)):
            over_threshold = over_threshold_list[j]
            print('over_threshold=', over_threshold)
            tarmaqLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)

            tarmaq_version = base_tarmaq_version + str(threshold)
            tarmaq_list = lowFreqTarmaqContextsResultDao.query_by(git_name, tarmaq_version)
            total_count = 0
            no_recommend = 0
            tarmaq_commit_th_target_minus_one = {}
            for doc in tarmaq_list:
                commit_th = doc['commit_th']
                cur_expected_validate_length = doc['cur_expected_validate_length']
                if doc['predict_result']['target'] == -1:
                    if commit_th in tarmaq_commit_th_target_minus_one:
                        tarmaq_commit_th_target_minus_one[commit_th] += 1
                    else:
                        tarmaq_commit_th_target_minus_one[commit_th] = 1
                else:
                    total_count += 1
                    if len(doc['predict_result']['topKList']) == 0:
                        no_recommend += 1
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
            print(git_name, total_count, no_recommend)
            # print('tarmaq contexts_component_summary', tarmaq_count)
            tarmaq_recall = tarmaqLowFreqTrace.contexts_component_summary(k_list, over_threshold=over_threshold)


if __name__ == '__main__':
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    sub_version = '_1_20_paper_'
    tarmaq_sub_version = '_1_30_fix_final_'
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        cbow_version_list = get_git_name_version(git_name, 'normal')
        subword_version_list = get_git_name_version(git_name, 'subword')
        base_tarmaq_version = git_name + tarmaq_sub_version
        main(git_name, base_tarmaq_version)




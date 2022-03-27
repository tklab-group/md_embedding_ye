import sys

sys.path.append('../')
from data.younger_trace import YoungerTrace
from data.mongo import YoungerTarmaqContextsResultDao
from data.mongo import YoungerContextsResultDao


if __name__ == '__main__':
    # tarmaq_version = 'tomcat_v1_younger_2'
    tarmaq_version = 'tomcat_v4_younger_4'
    # cbow_version = '2021_12_28_normal_tomcat_9_younger_2022_1_1'
    # cbow_version = '2021_12_30_normal_tomcat_1_younger_2022_1_4'
    cbow_version = '2021_1_2_tsubame_sub_word_tomcat_1_younger_2022_1_4'
    git_name = 'tomcat'
    k_list = [1, 5, 10, 15, 20]

    youngerTarmaqContextsResultDao = YoungerTarmaqContextsResultDao()
    youngerContextsResultDao = YoungerContextsResultDao()

    tarmaqYoungerTrace = YoungerTrace(all_id_to_word={})
    cbowYoungerTrace = YoungerTrace(all_id_to_word={})

    tarmaq_list = youngerTarmaqContextsResultDao.query_by(git_name, tarmaq_version)
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
        tarmaqYoungerTrace.save_contexts_component_predict({
            'contexts_component': {
                'new_rate': doc['new_rate'],
                'new_list': doc['new_list'],
                'old_rate': doc['old_rate'],
                'old_list': doc['old_list'],
                'younger_rate': doc['younger_rate'],
                'younger_list': doc['younger_list'],
            },
            'predict_result': {
                'target': doc['predict_result']['target'],
                'top100_aq': doc['predict_result']['topKList']
            },
            'commit_th': doc['commit_th']
        })
    print('tarmaq contexts_component_summary', tarmaq_count)
    tarmaq_recall = tarmaqYoungerTrace.contexts_component_summary(k_list)
    # print('tarmaq_count', tarmaq_count)
    print()

    cbow_list = youngerContextsResultDao.query_by(git_name, cbow_version)
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
        cbowYoungerTrace.save_contexts_component_predict({
            'contexts_component': {
                'new_rate': doc['new_rate'],
                'new_list': doc['new_list'],
                'old_rate': doc['old_rate'],
                'old_list': doc['old_list'],
                'younger_rate': doc['younger_rate'],
                'younger_list': doc['younger_list'],
            },
            'predict_result': doc['predict_result'],
            'commit_th': doc['commit_th']
        })
    # test
    print('tarmaq_commit_th_target_minus_one', len(tarmaq_commit_th_target_minus_one))
    # for commit_th in tarmaq_commit_th_target_minus_one:
    #     print(commit_th, tarmaq_commit_th_target_minus_one[commit_th])
    print('cbow_commit_th_target_minus_one', len(cbow_commit_th_target_minus_one))
    # for commit_th in cbow_commit_th_target_minus_one:
    #     print(commit_th, cbow_commit_th_target_minus_one[commit_th])
    for commit_th in tarmaq_commit_th_target_minus_one:
        if tarmaq_commit_th_target_minus_one[commit_th] != cbow_commit_th_target_minus_one[commit_th]:
            print(commit_th, tarmaq_commit_th_target_minus_one[commit_th], cbow_commit_th_target_minus_one[commit_th])

    print('cbow contexts_component_summary', cbow_count)
    cbow_recall = cbowYoungerTrace.contexts_component_summary(k_list)
    # print('cbow_count', cbow_count)
    print()

    print('tarmaq_recall new', tarmaq_recall['new'])
    print('tarmaq_recall old', tarmaq_recall['old'])
    print('tarmaq_recall younger', tarmaq_recall['younger'])

    print('cbow_recall new', cbow_recall['new'])
    print('cbow_recall old', cbow_recall['old'])
    print('cbow_recall younger', cbow_recall['younger'])

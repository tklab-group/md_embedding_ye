import sys

sys.path.append('../')
from data.younger_trace import YoungerTrace
from data.mongo import YoungerTarmaqResultDao
from data.mongo import YoungerResultDao


if __name__ == '__main__':
    tarmaq_version = 'tomcat_v3_younger_3'
    # cbow_version = '2021_12_28_normal_tomcat_9_younger_1'
    cbow_version = '2021_12_30_normal_tomcat_2_younger_2022_1_4'
    # cbow_version = '2021_1_2_tsubame_sub_word_tomcat_1_younger_2022_1_4'
    git_name = 'tomcat'
    k_list = [1, 5, 10, 15, 20]

    youngerTarmaqResultDao = YoungerTarmaqResultDao()
    youngerResultDao = YoungerResultDao()

    tarmaqYoungerTrace = YoungerTrace(all_id_to_word={})
    cbowYoungerTrace = YoungerTrace(all_id_to_word={})

    tarmaq_list = youngerTarmaqResultDao.query_by(git_name, tarmaq_version)
    tarmaq_count = 0
    tarmaq_commit_th_list = []
    for doc in tarmaq_list:
        tarmaq_count += 1
        tarmaqYoungerTrace.save_target_younger_predict({
            'target': doc['predict_result']['target'],
            'top100_aq': doc['predict_result']['topKList']
        })
        tarmaq_commit_th_list.append(doc['commit_th'])
    tarmaq_recall = tarmaqYoungerTrace.target_younger_summary(k_list)
    # print('tarmaq_count', tarmaq_count, tarmaq_commit_th_list)
    print('tarmaq_count', tarmaq_count)

    cbow_list = youngerResultDao.query_by(git_name, cbow_version)
    cbow_count = 0
    cbow_commit_th_list = []
    for doc in cbow_list:
        cbow_count += 1
        cbow_commit_th_list.append(doc['commit_th'])
        cbowYoungerTrace.save_target_younger_predict(doc['predict_result'])
    cbow_recall = cbowYoungerTrace.target_younger_summary(k_list)
    # print('cbow_count', cbow_count, cbow_commit_th_list)
    print('cbow_count', cbow_count)

    print('tarmaq_recall', tarmaq_recall)
    print('cbow_recall', cbow_recall)


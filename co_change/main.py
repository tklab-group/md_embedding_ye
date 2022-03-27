import sys

sys.path.append('../')
from data.util import get_co_change
from co_change.eval import Evaluation
from co_change.handle_target import HandleTarget


def eval_top_k(k, co_change, is_consider_new_file=True):
    evaluation = Evaluation(co_change)
    # micro_recall, macro_recall, mrr, f_mrr = evaluation.validate(k, is_consider_new_file)
    # print('k %d | new file %d | micro recall %.2f | macro recall %.2f | mrr %.2f | f_mrr %.2f'
    #           % (k, is_consider_new_file, micro_recall, macro_recall, mrr, f_mrr))
    micro_recall, macro_recall = evaluation.validate(k, is_consider_new_file)
    # print('k %d | new file %d | micro recall %.2f | macro recall %.2f'
    #           % (k, is_consider_new_file, micro_recall, macro_recall))
    print('k %d | micro recall %.2f | macro recall %.2f'
          % (k, micro_recall, macro_recall))


if __name__ == '__main__':
    git_name = 'tomcat_false'
    # git_name = 'LCExtractor_true'
    handleTarget = HandleTarget('tomcat', 1000)
    co_change = get_co_change(git_name)
    co_change = handleTarget.filter(co_change, False)
    # k = 1
    # for i in range(len(co_change)):
    #     print(co_change[i]['list'])
    # evaluation = Evaluation(co_change)
    # Recall, MRR, F_MRR = evaluation.validate(k)
    # print('k Recall, MRR, F_MRR', k, round(Recall, 2), round(MRR, 2), round(F_MRR, 2))
    print('tomcat false:')
    eval_top_k(1, co_change)
    eval_top_k(1, co_change, False)
    eval_top_k(5, co_change)
    eval_top_k(5, co_change, False)
    eval_top_k(10, co_change)
    eval_top_k(10, co_change, False)
    eval_top_k(20, co_change)
    eval_top_k(20, co_change, False)

    print('tomcat true:')
    git_name = 'tomcat_true'
    handleTarget = HandleTarget('tomcat', 1000)
    co_change = get_co_change(git_name)
    co_change = handleTarget.filter(co_change, True)
    eval_top_k(1, co_change)
    eval_top_k(1, co_change, False)
    eval_top_k(5, co_change)
    eval_top_k(5, co_change, False)
    eval_top_k(10, co_change)
    eval_top_k(10, co_change, False)
    eval_top_k(20, co_change)
    eval_top_k(20, co_change, False)


import sys
sys.path.append('../')
from data.util import get_co_change
from co_change.eval import Evaluation
from co_change.handle_target import HandleTarget


class AnaTarmaq:
    def __init__(self,
                 git_name,
                 expected_validate_length,
                 most_recent):
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent

        handleTarget = HandleTarget(git_name, expected_validate_length, most_recent)
        git_name_false = git_name + '_false'
        if most_recent > 0:
            git_name_false += '_' + str(most_recent)
        co_change_false = get_co_change(git_name_false)
        co_change_false = handleTarget.filter(co_change_false, False)
        self.co_change_false = co_change_false

        handleTarget = HandleTarget(git_name, expected_validate_length, most_recent)
        git_name_true = git_name + '_true'
        if most_recent > 0:
            git_name_true += '_' + str(most_recent)
        co_change_true = get_co_change(git_name_true)
        co_change_true = handleTarget.filter(co_change_true, True)
        self.co_change_true = co_change_true

    def getTopK(self, topKList, k):
        if len(topKList) <= k:
            return topKList
        return topKList[0: k]

    def ana(self, is_fix):
        if is_fix:
            co_change = self.co_change_true
        else:
            co_change = self.co_change_false
        top_k_empty_count = 0
        target_new_count = 0
        max_top_k = 0

        for i in range(len(co_change)):
            pair_list = co_change[i]['list']
            commit_th = i
            for j in range(len(pair_list)):
                pair = pair_list[j]
                # contexts = pair['contexts']
                target = pair['target']
                topKList = pair['topKList']
                max_top_k = max(len(topKList), max_top_k)
                if len(topKList) == 0:
                    top_k_empty_count += 1
                    if target == -1:
                        target_new_count += 1
        print('is_fix', is_fix)
        print('top_k_empty_count', top_k_empty_count)
        print('target_new_count', target_new_count)
        print(target_new_count / top_k_empty_count)
        print('max_top_k', max_top_k)


if __name__ == '__main__':
    git_name_ = 'tomcat'
    expected_validate_length_ = 1000
    most_recent_ = 5000
    anaTarmaq = AnaTarmaq(git_name_, expected_validate_length_, most_recent_)
    anaTarmaq.ana(True)
    anaTarmaq.ana(False)

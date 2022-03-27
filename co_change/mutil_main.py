import sys

sys.path.append('../')
from data.util import get_co_change
from co_change.eval import Evaluation
from co_change.main import eval_top_k
from data.repo_data import get_repo_data
from co_change.handle_target import HandleTarget
from data.data_store import DataStore
from co_change.eval import Evaluation


if __name__ == '__main__':
    repo_data = get_repo_data()
    dataStore = DataStore()
    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        print(git_name, expected_validate_length, most_recent)
        is_fix = False
        evaluation = Evaluation(git_name, expected_validate_length, most_recent, is_fix)
        # evaluation.summary(False)
        evaluation.summary(True)

        # is_fix = True
        # evaluation = Evaluation(git_name, expected_validate_length, most_recent, is_fix)
        # # evaluation.summary(False)
        # evaluation.summary(True)
        # print()

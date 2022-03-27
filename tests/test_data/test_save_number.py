import sys
sys.path.append('../../')

from common.util import save_data, load_data
import os
from data.util import load_predict_result, save_predict_result

if __name__ == '__main__':
    git_name = 'test'
    version = 'test_version'
    save_predict_result(git_name, version, 1)
    predict_result = load_predict_result(git_name, version)
    print(predict_result)

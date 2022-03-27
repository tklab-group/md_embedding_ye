import sys
sys.path.append('../../')
import time
import datetime
from data.data_loader import DataLoader
from data.mode_enum import Mode

git_name = 'tomcat'
# git_name = 'LCExtractor'
expected_validate_length = 1000
data_loader = DataLoader(git_name=git_name, expected_validate_length=expected_validate_length, mode=Mode.NORMAL)
data_loader.debug_info()



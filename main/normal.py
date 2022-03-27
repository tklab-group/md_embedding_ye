import sys
sys.path.append('../')
import torch
import time
from model.main import Main
from data.mode_enum import Mode
from data.data_store import DataStore

if __name__ == '__main__':
    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    print('is_can_cuda', is_can_cuda)

    # git_name = 'tomcat'
    git_name = 'LCExtractor'
    expected_validate_length = 19
    main = Main(git_name=git_name,
                expected_validate_length=expected_validate_length,
                mode=Mode.SUB_WORD_NO_FULL,
                max_epoch=10,
                dim=100,
                batch_size=32,
                is_sub_sampling=False,
                is_negative_sampling=True,
                dataStore=DataStore())
    # main.data_loader.debug_info()
    main.train(device=device)

    # main.eval(k=1)
    main.eval(k=1, is_consider_new_file=False)
    # main.eval(k=5)
    main.eval(k=5, is_consider_new_file=False)
    # main.eval(k=10)
    main.eval(k=10, is_consider_new_file=False)
    # main.eval(k=20)
    main.eval(k=20, is_consider_new_file=False)















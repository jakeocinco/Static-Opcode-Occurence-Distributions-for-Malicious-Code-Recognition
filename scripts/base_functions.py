import os

from config import *

def get_split_file_lists():
    # This functions gets all of the training sample op code lists and splits them into
    # two lists
    op_code_list = []
    clean_all = []
    infected_all = []

    for training_set in TRAINING_SAMPLES:
        _op_code_list = os.listdir(training_set['op_code_list_directory'])
        _op_code_list = list(
            map(
                lambda x: f"{training_set['op_code_list_directory']}/{x}",
                _op_code_list
            )
        )
        if '.DS_Store' in _op_code_list:
            _op_code_list.remove('.DS_Store')

        clean_all += list(
            filter(
                lambda x: 'clean' in x,
                _op_code_list
            )
        )
        infected_all += list(
            filter(
                lambda x: 'infect' in x,
                _op_code_list
            )
        )
        op_code_list += _op_code_list
    return op_code_list, clean_all, infected_all




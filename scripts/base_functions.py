import os

from op_codes import *
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


def _get_op_code_dictionary(op_codes):
    return {x: [] for x in op_codes}


def reduce_op_code_list_to_index_list(
        op_code_occurrence_list,
        op_codes
):
    file_operations = _get_op_code_dictionary(op_codes=op_codes)

    for line_index, line in enumerate(op_code_occurrence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]
        if operation in file_operations:
            file_operations[operation] += [line_index]

    return file_operations, len(op_code_occurrence_list)


def reduce_multiple_op_code_lists_to_index_lists(
        op_code_occurrence_list,
        op_code_options_dictionary
):

    file_operations = {
        key: _get_op_code_dictionary(op_codes=value) for key, value in op_code_options_dictionary.items()
    }

    for line_index, line in enumerate(op_code_occurrence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]

        for k in file_operations:
            if operation in file_operations[k]:
                file_operations[k][operation] += [line_index]

    for op_code_option in file_operations:
        if op_code_option in OP_CODE_CLUSTER:
            temp = {x: [] for x in set(OP_CODE_CLUSTER[op_code_option].values())}
            for op in file_operations[op_code_option]:
                temp[OP_CODE_CLUSTER[op_code_option][op]] += file_operations[op_code_option][op]

            for op in temp:
                temp[op] = sorted(temp[op])
            file_operations[op_code_option] = temp

    return file_operations, len(op_code_occurrence_list)




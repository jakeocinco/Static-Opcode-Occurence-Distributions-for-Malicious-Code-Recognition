import math
import numpy as np

from opcode_sets import *
from config import *


from sklearn import svm
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# from sklearn.decomposition import PCA
# from overwrite_dict import overwrite_dictionary_entries

MODELS = {
    'linear_svm': {'model': svm.SVC(kernel='linear'), 'scaler': None},
    'linear_svm_scaled': {'model': svm.SVC(kernel='linear'), 'scaler': StandardScaler()},
    'ridge': {'model': RidgeClassifier(), 'scaler': None},
    'ridge_scaled': {'model': RidgeClassifier(), 'scaler': StandardScaler()},
    'sgd': {'model': SGDClassifier(), 'scaler': StandardScaler()},
    'logistic': {'model': LogisticRegression(max_iter=1000),  'scaler': None},
    'logistic_scaled': {'model': LogisticRegression(max_iter=1000), 'scaler': StandardScaler()},
    'mlp': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': None},
    'mlp_scaled': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': StandardScaler()},
}

KL_METHODS = {
    'two_sided': [(lambda a, b: (KL(a, b) + KL(b, a)) / 2) for _ in range(2)],
    'x||dist': [(lambda a, b: KL(a, b)) for _ in range(2)],
    'dist||x': [(lambda a, b: KL(b, a)) for _ in range(2)],
    'x||dist_clean': [(lambda a, b: KL(a, b)), lambda a, b: 0],
    'dist||x_clean': [(lambda a, b: KL(b, a)), lambda a, b: 0],
    'x||dist_infected': [lambda a, b: 0, (lambda a, b: KL(a, b))],
    'dist||x_infected': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))],
    'two_sided_log': [(lambda a, b: (math.log(KL(a, b), 2) + math.log(KL(b, a), 2)) / 2) for _ in range(2)],
    'x||dist_log': [(lambda a, b: math.log(KL(a, b), 10)) for _ in range(2)],
    'dist||x_log': [(lambda a, b: math.log(KL(b, a), 10)) for _ in range(2)],
    'x||dist_clean_log': [(lambda a, b: math.log(KL(a, b), 10)), lambda a, b: 0],
    'dist||x_clean_log': [(lambda a, b: math.log(KL(b, a), 10)), lambda a, b: 0],
    'x||dist_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(a, b), 10))],
    'dist||x_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))]
}

def make_directory(path):
    # This is just a safe make directory function
    # In this case it does not matter if it already exists
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def find_all_sub_paths(path):
    # Split a long destionation down into a list of all paths between the end and root.
    # This is just used to make sure all folders are made if needed
    index = path.rfind("/")
    if index > 0:
        temp = path[:index]
        return [temp] + find_all_sub_paths(temp)
    return []


def create_all_possible_sub_directories(base_path, variants):

    directories = list(
        map(
            find_all_sub_paths,
            [variants[d]['options'][option]['bins'][b]['sub_path']
             for d in variants for option in variants[d]['options'] for b in variants[d]['options'][option]['bins']]
        )
    )
    directories = list(set([item for sublist in directories for item in sublist]))
    directories.sort(key=len)

    for directory in directories:
        make_directory(base_path + "/" + directory)


def KL(a, b):
    a = np.asarray(a, dtype=np.float) + .000001
    b = np.asarray(b, dtype=np.float) + .000001
    return max(.1, np.sum(a * np.log(a / b), 0))


def get_number_of_opcodes_tracked(op_code_options, option):
    # The size of the distribution changes depending on which method is used
    if option in OP_CODE_CLUSTER:
        return len(list(set(OP_CODE_CLUSTER[option].values())))
    return len(op_code_options[option])


def get_split_file_lists(sample_names=None):
    # This functions gets all of the training sample op code lists and splits them into
    # two lists
    if sample_names is None:
        sample_names = []

    op_code_list = []
    clean_all = []
    infected_all = []

    if not sample_names:
        sets = TRAINING_SAMPLES
    else:
        sets = []
        for sample in TRAINING_SAMPLES + TESTING_SAMPLES:
            if sample['name'] in sample_names:
                sets += [sample]

    for training_set in sets:
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




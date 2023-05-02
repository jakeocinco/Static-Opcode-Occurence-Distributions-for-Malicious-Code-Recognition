import numpy as np
import random
from PIL import Image
from time import time
from op_codes import *
import math
import matplotlib.pyplot as plt
import sys

from datetime import datetime
from copy import deepcopy
import pickle



import json
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from overwrite_dict import overwrite_dictionary_entries

from classifiers.MultiLayerClassifier import MultiLayerClassifier
from scipy.stats import mannwhitneyu, ranksums


def get_len(op_code_options, option):
    if option in OP_CODE_CLUSTER:
        return len(list(set(OP_CODE_CLUSTER[option].values())))
    return len(op_code_options[option])


def Mann_Whitney_U(a, b):
    _a = (a * 1000).astype(int)
    _b = (b * 1000).astype(int)

    a_expanded_counts = []
    b_expanded_counts = []
    for i in range(a.shape[0]):
        a_expanded_counts += [i + random.random() for _ in range(_a[i])]
        b_expanded_counts += [i + random.random() for _ in range(_b[i])]
    a_expanded_counts = (np.array(a_expanded_counts))
    b_expanded_counts = (np.array(b_expanded_counts))
    # print(a_expanded_counts.shape)
    # print(b_expanded_counts.shape)
    U, p = mannwhitneyu(a_expanded_counts, b_expanded_counts)
    return p


def KL(a, b):
    a = np.asarray(a, dtype=np.float) + .000001
    b = np.asarray(b, dtype=np.float) + .000001
    return max(.1, np.sum(a * np.log(a / b), 0))


def _get_image_to_distribution(path, method):
    # print(path)
    _image_data = np.load(path)
    # print(_image_data[0])
    # print(sum(_image_data[0]))
    return _image_data


def _get_variant_base_func(base_path, funcs_dictionary, bins, op_code_types, kl_func_dict, training_size, method, pruned_path):
    def func(sample_size, iteration):
        return {
            dist: {
                'op_code_types': {
                    op_code_type: {
                        'bins': {
                            bin_size: {
                                'image_data': np.zeros(
                                    (get_len(OP_CODE_DICT[pruned_path], op_code_type), bin_size)
                                ),
                                'clean_data': _get_image_to_distribution(
                                    f"{base_path}/{op_code_type}/op_codes/{dist}/{bin_size}_bins/{training_size}_samples/clean/{iteration}.npy",
                                    method
                                ),
                                'infected_data': _get_image_to_distribution(
                                    f"{base_path}/{op_code_type}/op_codes/{dist}/{bin_size}_bins/{training_size}_samples/infected/{iteration}.npy",
                                    method
                                ),
                                'kl_funcs': {
                                    kl_func: {
                                        'X': np.zeros((sample_size, 2 * get_len(OP_CODE_DICT[pruned_path], op_code_type) if method == 'jump' else bin_size * 2)),
                                        # 'X': np.zeros((sample_size, 2 * length)),
                                        'y': np.zeros(sample_size),
                                        'clean': kl_func_dict[kl_func][0],
                                        'infected': kl_func_dict[kl_func][1]
                                    } for kl_func in kl_func_dict
                                }
                            } for bin_size in bins
                        },
                    } for op_code_type in op_code_types
                },
                'distribution_func': funcs_dictionary[dist],
            } for dist in funcs_dictionary
        }.copy()

    return func


def _get_empty_data_sets(funcs_dictionary, bins, length):
    return {
        dist: {
            bin_size: np.zeros((length, bin_size)) for bin_size in bins
        } for dist in funcs_dictionary
    }


def get_distribution_data(
        op_code_directory,
        path,
        iteration,
        distribution_funcs,
        sample_size,
        bins,
        op_code_types,
        training_size,
        kl_funcs,
        method,
        pruned=False,
        random_seed=-1,
        update_text_prefix='',
        exclude_files=True
):
    if method not in ['jump', 'cumulative_share', 'share', 'inverse_jump']:
        raise Exception(f'Unknown Method | {method}')

    if random_seed > 0:
        random.seed(random_seed)

    pruned_path = 'base' if not pruned else method
    op_codes = []

    for op_code_type in op_code_types:
        op_codes += OP_CODE_DICT[pruned_path][op_code_type]
    op_codes = sorted(list(set(op_codes)))

    # length = get_len(OP_CODE_DICT[pruned_path], method)

    variant_func = _get_variant_base_func(
        path,
        distribution_funcs,
        bins,
        op_code_types=op_code_types,
        kl_func_dict=kl_funcs,
        training_size=training_size,
        method=method,
        pruned_path=pruned_path
    )

    arr = os.listdir(f"{op_code_directory}/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    # these are the images used in the distribution, don't want to fit or test off of those
    l = training_size * iteration if exclude_files else 0
    u = training_size * (iteration + 1) if exclude_files else 1
    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )[l:u]
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )[l:u]

    arr = list(
        filter(
            lambda x: not (x in infected or x in clean),
            arr
        )
    )

    random.shuffle(arr)
    sample_size = min(sample_size, len(arr))

    arr = arr[:sample_size]
    del clean
    del infected

    variants = variant_func(
        sample_size=sample_size,
        iteration=iteration
    )

    print(f"{update_text_prefix}[0/{sample_size}]        ")
    for file_index, file_name in enumerate(arr):
        # if ((file_index + 1) % 100) == 0:
        sys.stdout.write("\033[F")
        print(f"{update_text_prefix}[{file_index + 1}/{sample_size}]")

        with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

        combined_file_operations, line_index = reduce_op_code_list_to_index_list(
            file_data,
            op_codes,
            # (None if (op_code_type not in OP_CODE_CLUSTER) else OP_CODE_CLUSTER[op_code_type])
        )

        for op_code_type in op_code_types:
            if op_code_type in OP_CODE_CLUSTER:
                enum_ops = sorted(list(set(OP_CODE_CLUSTER[op_code_type].values())))
                file_operations = {k: [] for k in enum_ops}

                for op in OP_CODE_DICT[pruned_path][op_code_type]:
                    if op in combined_file_operations:
                        file_operations[OP_CODE_CLUSTER[op_code_type][op]] += combined_file_operations[op]

                for op in enum_ops:
                    file_operations[op] = sorted(file_operations[op])
            else:
                enum_ops = sorted(OP_CODE_DICT[pruned_path][op_code_type])
                file_operations = { k: v for k, v in combined_file_operations.items() if k in enum_ops}

            length = get_len(OP_CODE_DICT[pruned_path], op_code_type)
            current_images = _get_empty_data_sets(
                funcs_dictionary=distribution_funcs,
                bins=bins,
                length=length
            )

            for i, op in enumerate(enum_ops):
                file_operations[op] = sorted(file_operations[op])
                if method in ['jump', 'inverse_jump']:
                    number_of_operation_instances = len(file_operations[op])

                    if number_of_operation_instances > 1:
                        for jump_index in range(number_of_operation_instances - 1):
                            jump = file_operations[op][jump_index + 1] - file_operations[op][jump_index]

                            for d in variants:
                                for b in variants[d]['op_code_types'][op_code_type]['bins']:
                                    mapped_jump = variants[d]['distribution_func'](jump, b)
                                    if mapped_jump < 1:
                                        key = int((mapped_jump * b) // 1)
                                        current_images[d][b][i, key] += 1
                elif method == 'cumulative_share':
                    for op_line in file_operations[op]:
                        value = op_line / line_index

                        for d in variants:
                            for b in variants[d]['op_code_types'][op_code_type]['bins']:
                                if value < 1:
                                    key = int(value * b)
                                    current_images[d][b][i, key:] += 1
                    for d in variants:
                        for b in variants[d]['op_code_types'][op_code_type]['bins']:
                            current_images[d][b][i, :] += 1
                elif method == 'share':
                    for op_line in file_operations[op]:
                        value = op_line / line_index

                        for d in variants:
                            for b in variants[d]['op_code_types'][op_code_type]['bins']:
                                if value < 1:
                                    key = int(value * b)
                                    current_images[d][b][i, key] += 1
                    for d in variants:
                        for b in variants[d]['op_code_types'][op_code_type]['bins']:
                            current_images[d][b][i, :] += 1

            if method in ['jump']:
                for d in variants:
                    for b in variants[d]['op_code_types'][op_code_type]['bins']:
                        for i in range(length):
                            current_images[d][b][i, :] += 1
                            current_images[d][b][i, :] *= 1 / (sum(current_images[d][b][i, :]))
            elif method in ['cumulative_share', 'share', 'inverse_jump']:
                for d in variants:
                    for b in variants[d]['op_code_types'][op_code_type]['bins']:
                        for i in range(b):
                            current_images[d][b][:, i] *= (1 / max(sum(current_images[d][b][:, i]), 0.001))
                        # for i in range(length):
                        #     value = sum(current_images[d][b][i, :])
                        #     current_images[d][b][i, :] *= 1 / (value if value > 0 else .001)

            for d in variants:
                for b in variants[d]['op_code_types'][op_code_type]['bins']:
                    for kl_func in variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs']:

                        if method == 'jump':
                            for r in range(length):
                                variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, 2 * r] = \
                                    variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['clean'](
                                        current_images[d][b][r],
                                        variants[d]['op_code_types'][op_code_type]['bins'][b]['clean_data'][r]
                                    )

                                variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, (2 * r) + 1] = \
                                    variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['infected'](
                                        current_images[d][b][r],
                                        variants[d]['op_code_types'][op_code_type]['bins'][b]['infected_data'][r]
                                    )
                        elif method in ['share', 'cumulative_share', 'inverse_jump']:
                            for r in range(current_images[d][b].shape[1]):
                                if sum(current_images[d][b][:, r]) == 0:
                                    current_images[d][b][:, r] += 1 / current_images[d][b].shape[0]

                                variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, 2 * r] = \
                                    variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['clean'](
                                        current_images[d][b][:, r],
                                        variants[d]['op_code_types'][op_code_type]['bins'][b]['clean_data'][:, r]
                                    )

                                variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, (2 * r) + 1] = \
                                    variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['infected'](
                                        current_images[d][b][:, r],
                                        variants[d]['op_code_types'][op_code_type]['bins'][b]['infected_data'][:, r]
                                    )

                        # variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] = \
                        #     variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] \
                        #     / (max(variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index]) + 1)

                        variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['y'][file_index] = 'clean' in file_name

    # Remove lambdas so we can pickle this dictionary
    for d in variants:
        variants[d].pop('distribution_func')
        for op_type in variants[d]['op_code_types']:
            for b in variants[d]['op_code_types'][op_type]['bins']:
                for kl_func in variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs']:
                    variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func].pop('clean')
                    variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func].pop('infected')
                    # print(
                    #     variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func]['X'].shape
                    # )

    sys.stdout.write("\033[F")
    return variants


def train_data_on_models(
        variants,
        result_json,
        iteration,
        distribution_funcs,
        models,
        bins,
        op_code_types,
        kl_funcs,
        test_size=100
):
    top_results = {k / (test_size * 1000): {
        'string': '',
        'model': None,
        'file_name': ''
    } for k in range(20)}

    result_json.update({iteration: {}})

    for d in variants:
        if d in distribution_funcs:
            result_json[iteration].update({d: {}})
            for op_type in variants[d]['op_code_types']:
                if op_type in op_code_types:
                    result_json[iteration][d].update({op_type: {}})
                    for b in variants[d]['op_code_types'][op_type]['bins']:
                        if b in bins:
                            result_json[iteration][d][op_type].update({b: {}})

                            for kl_func in variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs']:
                                if kl_func in kl_funcs:
                                    result_json[iteration][d][op_type][b].update({kl_func: {}})

                                    for model_ in models:
                                        X_test = variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func]['X'][-test_size:]
                                        y_test = variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func]['y'][-test_size:].reshape(-1)

                                        y_test = y_test.astype(int)
                                        X = variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func]['X'][:-test_size]
                                        y = variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func]['y'][:-test_size]
                                        y = y.astype(int)

                                        scaler = deepcopy(
                                            models[model_]['scaler']
                                        )
                                        model = deepcopy(
                                            models[model_]['model']
                                        )

                                        if scaler is not None:
                                            scaler.fit(X)
                                            X = scaler.transform(X)
                                            X_test = scaler.transform(X_test)

                                        if 'unsupervised' in models[model_]:
                                            model.fit(
                                                np.concatenate((X, X_test))
                                            )
                                            results = model.labels_[-test_size:]
                                        else:
                                            if 'plot' in models[model_] and models[model_]['plot']:
                                                model.fit(X, y, X_test=X_test, y_test=y_test)
                                            else:
                                                model.fit(X, y)

                                            # Clean is 1, malicious is 0
                                            results = np.asarray(model.predict(X_test))

                                        recall = sum((results - y_test) > 0)

                                        accuracy = (sum(results == y_test) / test_size) + (random.random() / 1000)

                                        if 'unsupervised' in models[model_]:
                                            if accuracy < .5:
                                                accuracy = 1 - accuracy + .01
                                                recall = sum((y_test - results) > 0)

                                        result_json[iteration][d][op_type][b][kl_func].update({model_: (sum(results == y_test) / test_size)})

                                        min_acc = sorted(top_results)[0]
                                        if accuracy > min_acc:
                                            top_results.update({accuracy: {
                                                'string': f"{d}, {op_type}, {b}, {kl_func}, {model_}, missed malicious: {recall / test_size}",
                                                'file_name': f'./models/{op_type}_{model_}_{b}_{kl_func}_{iteration}.pickle',
                                                'model': model
                                            }})
                                            top_results.pop(min_acc)

    print(f"Top Results {iteration}...")
    for k in sorted(top_results.keys()):
        if k > .5:
            print(f"\t{int(k * 100) / 100}, {top_results[k]['string']}")
            file_name = top_results[k]['file_name']
            top_model = top_results[k]['model']

            pickle.dump(top_model, open(file_name, "wb"))


def compare_distribution_data(
        op_code_directory,
        path,
        iterations,
        distribution_funcs,
        sample_sizes,
        models,
        bins,
        op_code_types,
        training_size,
        kl_funcs,
        method,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}

    for sample_size in sample_sizes:

        for iteration in iterations:

            f = f'./data/{method}_{iteration}_{sample_size}_{random_seed}.npy'
            if pruned:
                f = f'./data/pruned_{method}_{iteration}_{sample_size}_{random_seed}.npy'

            # variants = np.load(
            #     f, allow_pickle=True
            # ).item()

            variants = get_distribution_data(
                op_code_directory=op_code_directory,
                path=path,
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                sample_size=sample_size,
                bins=bins,
                op_code_types=op_code_types,
                training_size=training_size,
                kl_funcs=kl_funcs,
                method=method,
                pruned=pruned,
                random_seed=random_seed
            )

            np.save(
                f,
                variants
            )
            # np.save(
            #     f,
            #     overwrite_dictionary_entries(n=variants, o=_variants)
            # )

            train_data_on_models(
                variants=variants,
                result_json=result_json,
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                models=models,
                bins=bins,
                op_code_types=op_code_types,
                kl_funcs=kl_funcs,
                test_size=test_size
            )

    return result_json


def combine_opcode_sets(
        op_code_directory,
        path,
        iterations,
        distribution_funcs,
        sample_sizes,
        models,
        bins,
        op_code_type_sets,
        training_size,
        kl_funcs,
        method,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}

    for sample_size in sample_sizes:

        for iteration in iterations:

            current_var = {}
            f = f'./data/{method}_{iteration}_{sample_size}_{random_seed}.npy'
            if pruned:
                f = f'./data/pruned_{method}_{iteration}_{sample_size}_{random_seed}.npy'

            variants = np.load(
                f, allow_pickle=True
            ).item()

            for d in distribution_funcs:
                current_var.update({d: {'op_code_types': {}}})

                for op_set in op_code_type_sets:
                    op_key = '_'.join(op_set)

                    current_var[d]['op_code_types'].update({op_key: {'bins': {}}})
                    for b in bins:
                        current_var[d]['op_code_types'][op_key]['bins'].update({b: {'kl_funcs': {}}})
                        for kl in kl_funcs:
                            current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'].update(
                                {kl: {'X': None, 'y': None}}
                            )
                    for op_type in op_set:
                        for b in bins:
                            for kl in kl_funcs:
                                if current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'] is None:
                                    current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'] = \
                                        deepcopy(variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl]['X'])
                                    current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['y'] = \
                                        deepcopy(variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl]['y'])
                                else:
                                    current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'] = np.concatenate(
                                        (current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'],
                                        variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl]['X']),
                                        axis=1

                                    )
                                    if sum(current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['y'] ==
                                          variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl]['y']) != \
                                            current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'].shape[0]:
                                        raise Exception(f"SETS DON'T MATCH - {iteration}, {op_set}, {kl}")
                                    # print(current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'].shape)

            train_data_on_models(
                variants=current_var,
                result_json=result_json,
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                models=models,
                bins=bins,
                op_code_types=['_'.join(op_set) for op_set in op_code_type_sets],
                kl_funcs=kl_funcs,
                test_size=test_size
            )
    return result_json


def combine_opcode_methods(
        op_code_directory,
        path,
        iterations,
        distribution_funcs,
        sample_sizes,
        models,
        bins,
        op_code_set,
        training_size,
        kl_funcs,
        methods,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}

    for sample_size in sample_sizes:

        for iteration in iterations:

            current_var = {}
            op_key = '_'.join(methods)

            for d in distribution_funcs:
                current_var.update({d: {'op_code_types': {}}})
                current_var[d]['op_code_types'].update({op_key: {'bins': {}}})
                for b in bins:
                    current_var[d]['op_code_types'][op_key]['bins'].update({b: {'kl_funcs': {}}})
                    for kl in kl_funcs:
                        current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'].update(
                            {kl: {'X': None, 'y': None}}
                        )

            for method in methods:
                f = f'./data/{method}_{iteration}_{sample_size}_{random_seed}.npy'
                if pruned:
                    f = f'./data/pruned_{method}_{iteration}_{sample_size}_{random_seed}.npy'

                variants = np.load(
                    f, allow_pickle=True
                ).item()

                for d in distribution_funcs:
                    for b in bins:
                        for kl in kl_funcs:
                            if current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'] is None:
                                current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'] = \
                                    deepcopy(variants[d]['op_code_types'][op_code_set]['bins'][b]['kl_funcs'][kl]['X'])
                                current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['y'] = \
                                    deepcopy(variants[d]['op_code_types'][op_code_set]['bins'][b]['kl_funcs'][kl]['y'])
                            else:
                                current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl][
                                    'X'] = np.concatenate(
                                    (current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'],
                                     variants[d]['op_code_types'][op_code_set]['bins'][b]['kl_funcs'][kl]['X']),
                                    axis=1

                                )
                                if sum(current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['y'] ==
                                       variants[d]['op_code_types'][op_code_set]['bins'][b]['kl_funcs'][kl]['y']) != \
                                        current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl][
                                            'X'].shape[0]:
                                    raise Exception(f"SETS DON'T MATCH - {iteration}, {op_code_set}, {kl}")
                                # print(current_var[d]['op_code_types'][op_key]['bins'][b]['kl_funcs'][kl]['X'].shape)

            train_data_on_models(
                variants=current_var,
                result_json=result_json,
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                models=models,
                bins=bins,
                op_code_types=[op_key],
                kl_funcs=kl_funcs,
                test_size=test_size
            )
    return result_json


def plot_comparisons(op_code_directory,
                     path,
                     iterations,
                     distribution_funcs,
                     sample_sizes,
                     bins,
                     op_code_type,
                     training_size,
                     kl_funcs,
                     method,
                     models=None,
                     pruned=False,
                     random_seed=-1,
                     test_size=100
                     ):
    for sample_size in sample_sizes:

        for iteration in iterations:

            variants = get_distribution_data(
                op_code_directory=op_code_directory,
                path=path,
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                sample_size=sample_size,
                models=models,
                bins=bins,
                op_code_type=op_code_type,
                training_size=training_size,
                kl_funcs=kl_funcs,
                method=method,
                pruned=pruned,
                test_size=test_size,
                random_seed=random_seed
            )

            for d in variants:
                for b in variants[d]['bins']:
                    for kl_func in variants[d]['bins'][b]['kl_funcs']:

                        X_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][-test_size:]
                        y_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][-test_size:].reshape(-1)
                        y_test = y_test.astype(int)
                        X = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][:-test_size]
                        y = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][:-test_size]
                        y = y.astype(int)

                        print(X.shape)

                        pca = PCA(n_components=2)
                        pca.fit(X)
                        X = pca.transform(X)
                        X_test = pca.transform(X_test)

                        x = [[], []]
                        #
                        # print(X)
                        # print(y)
                        for i in range(X.shape[0]):
                            x[y[i]] += [X[i]]
                            # print(y[i], X[i])
                        x[0] = np.array(x[0])
                        x[1] = np.array(x[1])

                        print(X.shape)
                        # print(x[0])
                        # print(x[1])
                        print(x[0].shape, x[1].shape)

                        plt.title(f'{method}, {d}, {b}, {kl_func}')
                        # plt.hist(x[0], alpha=0.5, label='Malicious')
                        # plt.hist(x[1], alpha=0.5, label='Benign')
                        plt.scatter(x[1][:, 0], x[1][:, 1], label='Benign', alpha=0.5)
                        plt.scatter(x[0][:, 0], x[0][:, 1], label='Malicious', alpha=0.5)
                        plt.legend()
                        plt.show()


if __name__ == "__main__":

    distributions = {
        'linear': lambda a, b: a / 1000,
        # 'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
        # 'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
        # 'threshold': lambda a, b: a / b
    }
    models = {
        # 'torch': {'model': MultiLayerClassifier(48, hidden_layers=[100, 200, 50]), 'scaler': None, 'plot': True},
        # 'linear_svm': {'model': svm.SVC(kernel='linear'), 'scaler': None},
        # 'linear_svm_scaled': {'model': svm.SVC(kernel='linear'), 'scaler': StandardScaler()},
        # 'ridge': {'model': RidgeClassifier(), 'scaler': None},
        # 'ridge_scaled': {'model': RidgeClassifier(), 'scaler': StandardScaler()},
        # 'sgd': {'model': SGDClassifier(), 'scaler': StandardScaler()},
        # 'logistic': {'model': LogisticRegression(max_iter=1000),  'scaler': None},
        # 'logistic_scaled': {'model': LogisticRegression(max_iter=1000), 'scaler': StandardScaler()},
        # 'mlp': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': None},
        'mlp_scaled': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': StandardScaler()},
        # 'mlp_scaled_big': {'model': MLPClassifier(hidden_layer_sizes=(500, 1000, 500, 250, 50), random_state=1, max_iter=600), 'scaler': StandardScaler()},
        # 'k_means': {'model': KMeans(n_clusters=2, random_state=0), 'scaler': None, 'unsupervised': True}
    }

    kls = {
        # 'two_sided': [(lambda a, b: (KL(a, b) + KL(b, a)) / 2) for _ in range(2)],
        # 'x||dist': [(lambda a, b: KL(a, b)) for _ in range(2)],
        # 'dist||x': [(lambda a, b: KL(b, a)) for _ in range(2)],
        # 'x||dist_clean': [(lambda a, b: KL(a, b)), lambda a, b: 0],
        # 'dist||x_clean': [(lambda a, b: KL(b, a)), lambda a, b: 0],
        # 'x||dist_infected': [lambda a, b: 0, (lambda a, b: KL(a, b))],
        # 'dist||x_infected': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))],
        # 'two_sided_log': [(lambda a, b: (math.log(KL(a, b), 2) + math.log(KL(b, a), 2)) / 2) for _ in range(2)],
        # 'x||dist_log': [(lambda a, b: math.log(KL(a, b), 10)) for _ in range(2)],
        'dist||x_log': [(lambda a, b: math.log(KL(b, a), 10)) for _ in range(2)],
        # 'x||dist_clean_log': [(lambda a, b: math.log(KL(a, b), 10)), lambda a, b: 0],
        # 'dist||x_clean_log': [(lambda a, b: math.log(KL(b, a), 10)), lambda a, b: 0],
        # 'x||dist_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(a, b), 10))],
        # 'dist||x_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))],
        # 'mann_whitney_u': [(lambda a, b: Mann_Whitney_U(a, b)) for _ in range(2)]
    }

    ops = ['infected', 'benign', 'union', 'intersection', 'disjoint'] #, 'ratio', 'ratio_a75', 'ratio_a25', 'malware_cluster', 'common_cluster']
    # PRUNED = True

    # [9, 1, 85, 83, 29]
    for random_seed in [85, 83]:
        for PRUNED in [False]:
            for method in ['share', 'cumulative_share']: #, 'share', 'cumulative_share', 'inverse_jump']: #
                pruned_path = 'pruned' if PRUNED else 'base'
                results_path = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset" \
                               f"/results/{method}_{random_seed}_{int(time())}.json"

                temp_results = compare_distribution_data(
                    op_code_directory="/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/",
                    path=f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset"
                         f"/op_code_distributions_samples/{method}/{pruned_path}/",
                    iterations=range(10),
                    distribution_funcs=distributions,
                    sample_sizes=[2500],
                    models=models,
                    bins=[100],
                    op_code_types=ops,
                    training_size=500,
                    kl_funcs=kls,
                    method=method,
                    pruned=PRUNED,
                    test_size=500,
                    random_seed=random_seed
                )
                # temp_results = combine_opcode_sets(
                #     op_code_directory="/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/",
                #     path=f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset"
                #          f"/op_code_distributions_samples/{method}/{pruned_path}/",
                #     iterations=range(10),
                #     distribution_funcs=distributions,
                #     sample_sizes=[2500],
                #     models=models,
                #     bins=[100],
                #     op_code_type_sets=[('ratio_a75', 'common_cluster'), ['common_cluster']],
                #     training_size=500,
                #     kl_funcs=kls,
                #     method=method,
                #     pruned=PRUNED,
                #     test_size=500,
                #     random_seed=random_seed
                # )
                # print(f"...{t.upper()}-{method.upper()}")
                # results.update()

                json_object = json.dumps({pruned_path: temp_results}, indent=4)

                # Writing to sample.json
                with open(results_path, "w") as outfile:
                    outfile.write(json_object)

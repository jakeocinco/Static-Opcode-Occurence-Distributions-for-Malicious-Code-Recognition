import random
import sys
import pickle

from time import time
from copy import deepcopy

import numpy as np

from base_functions import *


def _get_image_to_distribution(path):
    _image_data = np.load(path)
    return _image_data


def _get_variant_base_func(
        funcs_dictionary,
        bins,
        op_code_types,
        kl_func_dict,
        cumulative_distribution_samples_used,
        method,
        pruned_path
):
    def func(sample_size, iteration):
        return {
            dist: {
                'op_code_types': {
                    op_code_type: {
                        'bins': {
                            bin_size: {
                                'image_data': np.zeros(
                                    (get_number_of_opcodes_tracked(OP_CODE_DICT[pruned_path], op_code_type), bin_size)
                                ),
                                'clean_data': _get_image_to_distribution(
                                    f"{DISTRIBUTION_SAMPLE_PATH}/{method}/{pruned_path}/{op_code_type}/op_codes/{dist}/"
                                    f"{bin_size}_bins/{cumulative_distribution_samples_used}_samples/clean/{iteration}.npy"
                                ),
                                'infected_data': _get_image_to_distribution(
                                    f"{DISTRIBUTION_SAMPLE_PATH}/{method}/{pruned_path}/{op_code_type}/op_codes/{dist}/"
                                    f"{bin_size}_bins/{cumulative_distribution_samples_used}_samples/infected/{iteration}.npy"
                                ),
                                'kl_funcs': {
                                    kl_func: {
                                        'X': np.zeros((sample_size, 2 * get_number_of_opcodes_tracked(OP_CODE_DICT[pruned_path], op_code_type) if method == 'jump' else bin_size * 2)),
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
        iteration,
        distribution_funcs,
        sample_size,
        bins,
        op_code_types,
        cumulative_distribution_samples_used,
        kl_funcs,
        method,
        pruned=False,
        random_seed=-1,
        update_text_prefix='',
        exclude_files=True
):
    if method not in ['jump', 'cumulative_share', 'share']:
        raise Exception(f'Unknown Method | {method}')

    if random_seed > 0:
        random.seed(random_seed)

    pruned_path = 'base' if not pruned else method
    op_codes = []

    for op_code_type in op_code_types:
        op_codes += OP_CODE_DICT[pruned_path][op_code_type]
    op_codes = sorted(list(set(op_codes)))

    variant_func = _get_variant_base_func(
        distribution_funcs,
        bins,
        op_code_types=op_code_types,
        kl_func_dict=kl_funcs,
        cumulative_distribution_samples_used=cumulative_distribution_samples_used,
        method=method,
        pruned_path=pruned_path
    )

    arr, clean, infected = get_split_file_lists()

    # these are the images used in the distribution, don't want to fit or test off of those
    l = cumulative_distribution_samples_used * iteration
    u = cumulative_distribution_samples_used * (iteration + 1)
    clean = clean[l:u]
    infected = infected[l:u]

    # When testing on sets that do not include the training data we do not need to exclude any files
    if exclude_files:
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
        sys.stdout.write("\033[F")
        print(f"{update_text_prefix}[{file_index + 1}/{sample_size}]")

        with open(file_name) as file:
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

        combined_file_operations, line_index = reduce_op_code_list_to_index_list(
            file_data,
            op_codes
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

            length = get_number_of_opcodes_tracked(OP_CODE_DICT[pruned_path], op_code_type)
            current_images = _get_empty_data_sets(
                funcs_dictionary=distribution_funcs,
                bins=bins,
                length=length
            )

            for i, op in enumerate(enum_ops):
                file_operations[op] = sorted(file_operations[op])
                if method in ['jump']:
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
            elif method in ['cumulative_share', 'share']:
                for d in variants:
                    for b in variants[d]['op_code_types'][op_code_type]['bins']:
                        for i in range(b):
                            current_images[d][b][:, i] *= (1 / max(sum(current_images[d][b][:, i]), 0.001))

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
                        elif method in ['share', 'cumulative_share']:
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

                        variants[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['y'][file_index] = 'clean' in file_name

    # Remove lambdas so we can pickle this dictionary to save data
    for d in variants:
        variants[d].pop('distribution_func')
        for op_type in variants[d]['op_code_types']:
            for b in variants[d]['op_code_types'][op_type]['bins']:
                for kl_func in variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs']:
                    variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func].pop('clean')
                    variants[d]['op_code_types'][op_type]['bins'][b]['kl_funcs'][kl_func].pop('infected')

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


def get_test_accuracy_of_process(
        iterations,
        sample_sizes,
        models,
        bins,
        op_code_types,
        cumulative_distribution_samples_used,
        kl_funcs,
        method,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}
    distribution_funcs = {
        'linear': lambda a, b: a / 1000
    }

    for sample_size in sample_sizes:

        for iteration in iterations:

            f = f'./data/{method}_{iteration}_{sample_size}_{random_seed}.npy'
            if pruned:
                f = f'./data/pruned_{method}_{iteration}_{sample_size}_{random_seed}.npy'

            # variants = np.load(
            #     f, allow_pickle=True
            # ).item()

            variants = get_distribution_data(
                iteration=iteration,
                distribution_funcs=distribution_funcs,
                sample_size=sample_size,
                bins=bins,
                op_code_types=op_code_types,
                cumulative_distribution_samples_used=cumulative_distribution_samples_used,
                kl_funcs=kl_funcs,
                method=method,
                pruned=pruned,
                random_seed=random_seed
            )

            # np.save(
            #     f,
            #     variants
            # )
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


def get_test_accuracy_of_process_combined_op_code_sets(
        iterations,
        sample_sizes,
        models,
        bins,
        op_code_type_sets,
        cumulative_distribution_samples_used,
        kl_funcs,
        method,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}
    distribution_funcs = {
        'linear': lambda a, b: a / 1000
    }

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


def get_test_accuracy_of_process_combined_methods(
        iterations,
        sample_sizes,
        models,
        bins,
        op_code_set,
        cumulative_distribution_samples_used,
        kl_funcs,
        methods,
        random_seed,
        pruned=False,
        test_size=100
):
    result_json = {}
    distribution_funcs = {
        'linear': lambda a, b: a / 1000
    }

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


if __name__ == "__main__":


    models = ['mlp_scaled']
    kls = ['dist||x_log']

    ops = ['infected', 'benign', 'union', 'intersection', 'disjoint']

    sample_size = 75
    RANDOM_SEEDS = [9, 1, 85, 83]
    methods = ['jump'] # can also use 'share', 'cumulative_share'
    bins = [100]
    cumulative_distribution_samples_used = 500 # this is just used as a param to find the write distributions samples

    for random_seed in [9]:
        for pruned in [False]:
            for method in methods:
                results_file_name = f"{method}_{random_seed}_{int(time())}.json"
                pruned_path = 'base' if not pruned else method

                temp_results = get_test_accuracy_of_process(
                    iterations=range(1),
                    sample_sizes=[sample_size],
                    models={k:MODELS[k] for k in models},
                    bins=bins,
                    op_code_types=ops,
                    cumulative_distribution_samples_used=cumulative_distribution_samples_used,
                    kl_funcs={kl:KL_METHODS[kl] for kl in kls},
                    method=method,
                    pruned=pruned,
                    test_size=int(sample_size * .2),
                    random_seed=random_seed
                )

                json_object = json.dumps({pruned_path: temp_results}, indent=4)

                # Writing to sample.json
                with open(f"{RESULTS_BASE_PATH}/{results_file_name}", "w") as outfile:
                    outfile.write(json_object)

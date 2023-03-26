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
import dill

import json
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from classifiers.MultiLayerClassifier import MultiLayerClassifier
from scipy.stats import mannwhitneyu, ranksums


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


def _get_variant_base_func(base_path, funcs_dictionary, bins, length, kl_func_dict, training_size, method):
    def func(sample_size, iteration):
        return {
            dist: {
                'bins': {
                    bin_size: {
                        'image_data': np.zeros((length, bin_size)),
                        'clean_data': _get_image_to_distribution(
                            f"{base_path}/{dist}/{bin_size}_bins/{training_size}_samples/clean/{iteration}.npy",
                            method
                        ),
                        'infected_data': _get_image_to_distribution(
                            f"{base_path}/{dist}/{bin_size}_bins/{training_size}_samples/infected/{iteration}.npy",
                            method
                        ),
                        'kl_funcs': {
                            kl_func: {
                                'X': np.zeros((sample_size, 2 * length if method == 'jump' else bin_size * 2)),
                                # 'X': np.zeros((sample_size, 2 * length)),
                                'y': np.zeros(sample_size),
                                'clean': kl_func_dict[kl_func][0],
                                'infected': kl_func_dict[kl_func][1]
                            } for kl_func in kl_func_dict
                        }
                    } for bin_size in bins
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
        op_code_type,
        training_size,
        kl_funcs,
        method,
        pruned=False,
        random_seed=-1,
        update_text_prefix=''
):
    if method not in ['jump', 'cumulative_share', 'share', 'inverse_jump']:
        raise Exception(f'Unknown Method | {method}')

    if random_seed > 0:
        random.seed(random_seed)

    op_codes = OP_CODE_DICT['base' if not pruned else method][op_code_type]
    op_codes = sorted(op_codes)

    # print(op_codes.keys())
    if op_code_type in OP_CODE_CLUSTER:
        length = len(list(set(OP_CODE_CLUSTER[op_code_type].values())))
    else:
        length = len(op_codes)

    variant_func = _get_variant_base_func(
        path,
        distribution_funcs,
        bins,
        length=length,
        kl_func_dict=kl_funcs,
        training_size=training_size,
        method=method
    )

    arr = os.listdir(f"{op_code_directory}/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    # these are the images used in the distribution, don't want to fit or test off of those
    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )[(training_size * iteration):(training_size * (iteration + 1))]
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )[(training_size * iteration):(training_size * (iteration + 1))]

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

    print(f"{update_text_prefix}[0/{sample_size}]")
    for file_index, file_name in enumerate(arr):
        # if ((file_index + 1) % 100) == 0:
        sys.stdout.write("\033[F")
        print(f"{update_text_prefix}[{file_index + 1}/{sample_size}]")

        with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

        file_operations, line_index = reduce_op_code_list_to_index_list(
            file_data,
            op_codes,
            (None if (op_code_type not in OP_CODE_CLUSTER) else OP_CODE_CLUSTER[op_code_type])
        )

        enum_ops = op_codes
        if op_code_type in OP_CODE_CLUSTER:
            enum_ops = sorted(list(set(OP_CODE_CLUSTER[op_code_type].values())))

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
                            for b in variants[d]['bins']:
                                mapped_jump = variants[d]['distribution_func'](jump, b)
                                if mapped_jump < 1:
                                    key = int((mapped_jump * b) // 1)
                                    current_images[d][b][i, key] += 1
            elif method == 'cumulative_share':
                for op_line in file_operations[op]:
                    value = op_line / line_index

                    for d in variants:
                        for b in variants[d]['bins']:
                            if value < 1:
                                key = int(value * b)
                                current_images[d][b][i, key:] += 1
                for d in variants:
                    for b in variants[d]['bins']:
                        current_images[d][b][i, :] += 1
            elif method == 'share':
                for op_line in file_operations[op]:
                    value = op_line / line_index

                    for d in variants:
                        for b in variants[d]['bins']:
                            if value < 1:
                                key = int(value * b)
                                current_images[d][b][i, key] += 1
                for d in variants:
                    for b in variants[d]['bins']:
                        current_images[d][b][i, :] += 1

        if method in ['jump']:
            for d in variants:
                for b in variants[d]['bins']:
                    for i in range(length):
                        current_images[d][b][i, :] += 1
                        current_images[d][b][i, :] *= 1 / (sum(current_images[d][b][i, :]))
        elif method in ['cumulative_share', 'share', 'inverse_jump']:
            for d in variants:
                for b in variants[d]['bins']:
                    for i in range(b):
                        current_images[d][b][:, i] *= (1 / max(sum(current_images[d][b][:, i]), 0.001))
                    # for i in range(length):
                    #     value = sum(current_images[d][b][i, :])
                    #     current_images[d][b][i, :] *= 1 / (value if value > 0 else .001)

        for d in variants:
            for b in variants[d]['bins']:
                for kl_func in variants[d]['bins'][b]['kl_funcs']:

                    if method == 'jump':
                        for r in range(length):
                            variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, 2 * r] = \
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['clean'](
                                    current_images[d][b][r],
                                    variants[d]['bins'][b]['clean_data'][r]
                                )

                            variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, (2 * r) + 1] = \
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['infected'](
                                    current_images[d][b][r],
                                    variants[d]['bins'][b]['infected_data'][r]
                                )
                    elif method in ['share', 'cumulative_share', 'inverse_jump']:
                        for r in range(current_images[d][b].shape[1]):
                            if sum(current_images[d][b][:, r]) == 0:
                                current_images[d][b][:, r] += 1 / current_images[d][b].shape[0]

                            variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, 2 * r] = \
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['clean'](
                                    current_images[d][b][:, r],
                                    variants[d]['bins'][b]['clean_data'][:, r]
                                )

                            variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, (2 * r) + 1] = \
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['infected'](
                                    current_images[d][b][:, r],
                                    variants[d]['bins'][b]['infected_data'][:, r]
                                )

                    # variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] = \
                    #     variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] \
                    #     / (max(variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index]) + 1)

                    variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][file_index] = 'clean' in file_name

    # Remove lambdas so we can pickle this dictionary
    for d in variants:
        variants[d].pop('distribution_func')
        for b in variants[d]['bins']:
            for kl_func in variants[d]['bins'][b]['kl_funcs']:
                variants[d]['bins'][b]['kl_funcs'][kl_func].pop('clean')
                variants[d]['bins'][b]['kl_funcs'][kl_func].pop('infected')

    sys.stdout.write("\033[F")
    return variants


def compare_distribution_data(
        op_code_directory,
        path,
        iterations,
        distribution_funcs,
        sample_sizes,
        models,
        bins,
        op_code_type,
        training_size,
        kl_funcs,
        method,
        pruned=False,
        random_seed=-1,
        test_size=100
):
    for sample_size in sample_sizes:

        for iteration in iterations:

            # variants = get_distribution_data(
            #     op_code_directory=op_code_directory,
            #     path=path,
            #     iteration=iteration,
            #     distribution_funcs=distribution_funcs,
            #     sample_size=sample_size,
            #     bins=bins,
            #     op_code_type=op_code_type,
            #     training_size=training_size,
            #     kl_funcs=kl_funcs,
            #     method=method,
            #     pruned=pruned,
            #     random_seed=random_seed
            # )
            # np.save(
            #     f'./data/{iteration}_{sample_size}.npy',
            #     variants
            # )

            variants = np.load(
                f'./data/{iteration}_{sample_size}.npy',
                allow_pickle=True
            ).item()

            top_results = {k / (test_size * 1000): {
                'string': '',
                'model': None,
                'file_name': ''
            } for k in range(10)}

            result_json = {}
            result_json.update({iteration: {}})

            for d in variants:
                result_json[iteration].update({d: {}})

                for b in variants[d]['bins']:
                    result_json[iteration][d].update({b: {}})

                    for kl_func in variants[d]['bins'][b]['kl_funcs']:
                        if kl_func in kl_funcs:
                            result_json[iteration][d][b].update({kl_func: {}})

                            for model_ in models:
                                X_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][-test_size:]
                                y_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][-test_size:].reshape(-1)

                                y_test = y_test.astype(int)
                                X = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][:-test_size]
                                y = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][:-test_size]
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

                                result_json[iteration][d][b][kl_func].update({model_: (sum(results == y_test) / 100)})

                                min_acc = sorted(top_results)[0]
                                if accuracy > min_acc:
                                    top_results.update({accuracy: {
                                        'string': f"{d}, {b}, {kl_func}, {model_}, missed malicious: {recall / test_size}",
                                        'file_name': f'./models/{model_}_{b}_{kl_func}.pickle',
                                        'model': model
                                    }})
                                    top_results.pop(min_acc)

                        #     print(
                        #         f"\t{iteration}, {d}, {b}, {kl_func}, {model_}, {int(accuracy * 100) / 100}, missed malicious: {recall}"
                        #     )
                        # print()
            top_model = None
            file_name = ''
            print(f"Top Results {iteration}...")
            for k in sorted(top_results.keys()):
                if k > .5:
                    print(f"\t{int(k * 100) / 100}, {top_results[k]['string']}")
                    file_name = top_results[k]['file_name']
                    top_model = top_results[k]['model']

                    pickle.dump(top_model, open(file_name, "wb"))

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
        # 'torch': {'model': MultiLayerClassifier(48, hidden_layers=[100, 200, 50]), 'scaler': None},
        # 'linear_svm': {'model': svm.SVC(kernel='linear'), 'scaler': None},
        # 'linear_svm_scaled': {'model': svm.SVC(kernel='linear'), 'scaler': StandardScaler()},
        # 'ridge': {'model': RidgeClassifier(), 'scaler': None},
        # 'ridge_scaled': {'model': RidgeClassifier(), 'scaler': StandardScaler()},
        # 'sgd': {'model': SGDClassifier(), 'scaler': StandardScaler()},
        # 'logistic': {'model': LogisticRegression(max_iter=1000),  'scaler': None},
        # 'logistic_scaled': {'model': LogisticRegression(max_iter=1000), 'scaler': StandardScaler()},
        'mlp': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': None},
        'mlp_scaled': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': StandardScaler()},
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
        'two_sided_log': [(lambda a, b: (math.log(KL(a, b), 10) + math.log(KL(a, b), 10)) / 2) for _ in range(2)],
        # 'x||dist_log': [(lambda a, b: math.log(KL(a, b), 10)) for _ in range(2)],
        # 'dist||x_log': [(lambda a, b: math.log(KL(b, a), 10)) for _ in range(2)],
        # 'dist||x_clean_log': [(lambda a, b: math.log(KL(b, a), 10)), lambda a, b: 0],
        # 'x||dist_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(a, b), 10))],
        # 'dist||x_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))],
        # 'mann_whitney_u': [(lambda a, b: Mann_Whitney_U(a, b)) for _ in range(2)]
    }

    # ops = ['benign', 'infected', 'union', 'intersection', 'disjoint', 'ratio', 'ratio_a75']
    ops = ['common_cluster']
    # PRUNED = True

    for PRUNED in [False]:
        for method in ['jump']:  # , 'share', 'cumulative_share']: #, 'share', 'cumulative_share', 'inverse_jump']: #
            pruned_path = 'pruned' if PRUNED else 'base'
            results_path = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset" \
                           f"/results/{method}_{pruned_path}_{int(time())}.json"

            results = {}

            custom_op = {x: OP_CODE_DICT['base' if not PRUNED else method][x] for x in ops}
            for i, t in enumerate(custom_op):
                results.update({t: {}})
                if i > 0:
                    print(f"\n\n")
                print(f"{t.upper()}-{method.upper()}-{pruned_path.upper()} ...")
                # plot_comparisons(
                temp_results = compare_distribution_data(
                    op_code_directory="/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/",
                    path=f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset"
                         f"/op_code_distributions_samples/{method}/{pruned_path}/{t}/op_codes",
                    iterations=[5],
                    distribution_funcs=distributions,
                    sample_sizes=[2000],
                    models=models,
                    bins=[25, 100],
                    op_code_type=t,
                    training_size=500,
                    kl_funcs=kls,
                    method=method,
                    pruned=PRUNED,
                    test_size=100
                )
                print(f"...{t.upper()}-{method.upper()}")

                # results[t].update(temp_results)
                #
                # json_object = json.dumps(results, indent=4)
                #
                # # Writing to sample.json
                # with open(results_path, "w") as outfile:
                #     outfile.write(json_object)

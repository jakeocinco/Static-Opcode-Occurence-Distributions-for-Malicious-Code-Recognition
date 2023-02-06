import numpy as np
import random
from PIL import Image
from time import time
from op_codes import *
import math
import matplotlib.pyplot as plt

from datetime import datetime
from copy import deepcopy

import json
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def KL(a, b):
    a = np.asarray(a, dtype=np.float) + .00001
    b = np.asarray(b, dtype=np.float) + .00001
    return np.sum(a * np.log(a / b), 0)


def _get_image_to_distribution(path):
    _image_data = np.load(path)
    _image_data += 0.0001

    return _image_data


def _get_variant_base_func(base_path, funcs_dictionary, bins, length, models, kl_func_dict, training_size):

    def func(sample_size, iteration):
        return {
            dist: {
                'bins': {
                    bin_size: {
                        'image_data': np.zeros((length, bin_size)),
                        'clean_data': _get_image_to_distribution(
                            f"{base_path}/{dist}/{bin_size}_bins/{training_size}_samples/clean/{iteration}.npy"
                        ),
                        'infected_data': _get_image_to_distribution(
                            f"{base_path}/{dist}/{bin_size}_bins/{training_size}_samples/infected/{iteration}.npy"
                        ),
                        'kl_funcs': {
                            kl_func: {
                                'X': np.zeros((sample_size, 2 * length)),
                                'y': np.zeros(sample_size),
                                'clean': kl_func_dict[kl_func][0],
                                'infected': kl_func_dict[kl_func][1],
                                'models': {
                                    model: {
                                        'model': models[model],
                                        'accuracy': 0.0
                                    } for model in models
                                },
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


def compare_distribution_data(
        op_code_directory,
        path,
        iterations,
        distribution_funcs,
        sample_sizes,
        models,
        bins,
        op_codes,
        training_size,
        kl_funcs,
        random_seed=-1,
        is_jump=True
):


    if random_seed > 0:
        random.seed(random_seed)

    length = len(op_codes)
    variant_func = _get_variant_base_func(
        path,
        distribution_funcs,
        bins,
        length,
        kl_func_dict=kl_funcs,
        models=models,
        training_size=training_size
    )

    result_json = {}
    for sample_size in sample_sizes:

        for iteration in iterations:
            result_json.update({iteration: {}})

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
                    lambda x: not(x in infected or x in clean),
                    arr
                )
            )
            random.shuffle(arr)
            arr = arr[:sample_size]
            del clean
            del infected

            variants = variant_func(
                sample_size=sample_size,
                iteration=iteration
            )

            for file_index, file_name in enumerate(arr):

                row = np.array([0.0 for _ in range(length * 2)]).astype(np.float)

                with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
                    try:
                        file_data = str(file.read()).split()
                    except:
                        print(file_name)

                file_operations, line_index = reduce_op_code_list_to_index_list(
                    file_data,
                    op_codes
                )

                current_images = _get_empty_data_sets(
                    funcs_dictionary=distribution_funcs,
                    bins=bins,
                    length=length
                )

                for i, op in enumerate(op_codes):
                    if is_jump:
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
                        for d in variants:
                            for b in variants[d]['bins']:
                                current_images[d][b][i, :] += 1
                                current_images[d][b][i, :] *= 1 / (sum(current_images[d][b][i, :]))
                    else:
                        for op_line in file_operations[op]:
                            value = op_line / line_index

                            for d in variants:
                                for b in variants[d]['bins']:
                                    if value < 1:
                                        key = int(value * b)
                                        variants[d]['bins'][b]['image_data'][i, key:] += 1
                if not is_jump:
                    for d in variants:
                        for b in variants[d]['bins']:
                            for i in range(b):
                                current_images[d][b][:, i] *= (1 / max(sum(current_images[d][b][:, i]), 1))
                            for i in range(length):
                                current_images[d][b][i, :] *= (1 / max(max(current_images[d][b][i, :]), 1))

                for d in variants:
                    for b in variants[d]['bins']:
                        for kl_func in variants[d]['bins'][b]['kl_funcs']:

                            for r in range(length):
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, 2 * r] = \
                                    variants[d]['bins'][b]['kl_funcs'][kl_func]['clean'](
                                        variants[d]['bins'][b]['clean_data'][r],
                                        current_images[d][b][r]
                                    )

                                variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index, (2 * r) + 1] = \
                                    variants[d]['bins'][b]['kl_funcs'][kl_func]['infected'](
                                        variants[d]['bins'][b]['infected_data'][r],
                                        current_images[d][b][r]
                                    )

                            variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] = \
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index] \
                                / (max(variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][file_index]) + 1)

                            variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][file_index] = 'clean' in file_name

            top_results = {k / 100: '' for k in range(5)}

            print(f'-- {iteration} -- ')
            for d in variants:
                result_json[iteration].update({d: {}})

                for b in variants[d]['bins']:
                    result_json[iteration][d].update({b: {}})

                    for kl_func in variants[d]['bins'][b]['kl_funcs']:
                        result_json[iteration][d][b].update({kl_func: {}})

                        for model_ in variants[d]['bins'][b]['kl_funcs'][kl_func]['models']:

                            X_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][-100:]
                            y_test = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][-100:].reshape(-1)
                            y_test = y_test.astype(int)
                            X = variants[d]['bins'][b]['kl_funcs'][kl_func]['X'][:-100]
                            y = variants[d]['bins'][b]['kl_funcs'][kl_func]['y'][:-100]
                            y = y.astype(int)

                            scaler = deepcopy(
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['models'][model_]['model']['scaler']
                            )
                            model = deepcopy(
                                variants[d]['bins'][b]['kl_funcs'][kl_func]['models'][model_]['model']['model']
                            )

                            if scaler is not None:
                                scaler.fit(X)
                                X = scaler.transform(X)
                                X_test = scaler.transform(X_test)

                            model.fit(X, y)

                            results = np.asarray(model.predict(X_test))

                            accuracy = (sum(results == y_test) / 100) + (random.random() / 100)
                            result_json[iteration][d][b][kl_func].update({model_: (sum(results == y_test) / 100)})

                            min_acc = sorted(top_results)[0]
                            if accuracy > min_acc:
                                top_results.update({accuracy: f", {d}, {b}, {kl_func}, {model_}"})
                                top_results.pop(min_acc)

                            print(iteration, d, b, kl_func, model_, int(accuracy * 100) / 100)
                        print()

            for k in sorted(top_results.keys()):
                print(int(k * 100) / 100, top_results[k])

    return result_json


if __name__ == "__main__":
    results_path = f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/results/{int(time())}.json"

    distributions = {
        'linear': lambda a, b: a / 1000,
        # 'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
        # 'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
        # 'threshold': lambda a, b: a / b
    }
    models = {
        'linear_svm': {'model': svm.SVC(kernel='linear'), 'scaler': None},
        'linear_svm_scaled': {'model': svm.SVC(kernel='linear'), 'scaler': StandardScaler()},
        'ridge': {'model': RidgeClassifier(), 'scaler': None},
        'ridge_scaled': {'model': RidgeClassifier(), 'scaler': StandardScaler()},
        'sgd': {'model': SGDClassifier(), 'scaler': StandardScaler()},
        'logistic': {'model': LogisticRegression(max_iter=1000),  'scaler': None},
        'logistic_scaled': {'model': LogisticRegression(max_iter=1000), 'scaler': StandardScaler()},
        'mlp': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': None},
        'mlp_scaled': {'model': MLPClassifier(hidden_layer_sizes=(100, 200, 50), random_state=1, max_iter=300), 'scaler': StandardScaler()}
    }

    kls = {
        'two_sided': [(lambda a, b: (KL(a, b) + KL(b, a)) / 2) for _ in range(2)],
        'x||dist': [(lambda a, b: KL(a, b)) for _ in range(2)],
        'dist||x': [(lambda a, b: KL(b, a)) for _ in range(2)],
        'x||dist_clean': [(lambda a, b: KL(a, b)), lambda a, b: 0],
        'dist||x_clean': [(lambda a, b: KL(b, a)), lambda a, b: 0],
        'x||dist_infected': [lambda a, b: 0, (lambda a, b: KL(a, b))],
        'dist||x_infected': [lambda a, b: 0, (lambda a, b: KL(b, a))]
    }

    results = {}

    # custom_op = ['ratio', 'ratio_a25', 'ratio_a75']

    for t in OP_CODE_DICT:
        results.update({t: {}})
        print(f"\n\n{t}")
        temp_results = compare_distribution_data(
            op_code_directory="/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/",
            path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset"
                 f"/op_code_distributions_samples/share/{t}/op_codes",
            iterations=[4],
            distribution_funcs=distributions,
            sample_sizes=[1100],
            models=models,
            bins=[25, 100],
            op_codes=OP_CODE_DICT[t],
            training_size=500,
            kl_funcs=kls,
            is_jump=False
        )

        results[t].update(temp_results)

    json_object = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(results_path, "w") as outfile:
        outfile.write(json_object)


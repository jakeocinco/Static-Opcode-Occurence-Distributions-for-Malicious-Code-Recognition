from train_models import get_distribution_data, KL

import numpy as np
import math
import pickle

import random
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# random.seed(85)
SEEDS = [198, 598, 430, 206, 602, 992, 786, 527, 23, 371, 175, 291, 960, 706, 736, 750, 748, 247, 664, 999, 559, 570,
         584, 727, 257]

distributions = {
    'linear': lambda a, b: a / 1000,
}

SAMPLE_SIZE = 250
METHOD = "jump"
MALICIOUS_TRAIN_PATH = "/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/"
VIRUS_SHARE_000 = "/Volumes/T7/VirusShare/VirusShare_00000"
VIRUS_SHARE_005= "/Volumes/T7/VirusShare/VirusShare_00005"
VIRUS_SHARE_451= "/Volumes/T7/VirusShare/VirusShare_00451"
MAC_OS_PATH = "/Volumes/T7/MacOS/"
DISTRIBUTION_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples" \
                    f"/{METHOD}/base"
BINS = [100]

kls = {
    # 'two_sided': [(lambda a, b: (KL(a, b) + KL(b, a)) / 2) for _ in range(2)],
    # 'x||dist': [(lambda a, b: KL(a, b)) for _ in range(2)],
    # 'dist||x': [(lambda a, b: KL(b, a)) for _ in range(2)],
    # 'x||dist_clean': [(lambda a, b: KL(a, b)), lambda a, b: 0],
    # 'dist||x_clean': [(lambda a, b: KL(b, a)), lambda a, b: 0],
    # 'x||dist_infected': [lambda a, b: 0, (lambda a, b: KL(a, b))],
    # 'dist||x_infected': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))],
    # 'two_sided_log': [(lambda a, b: (math.log(KL(a, b), 10) + math.log(KL(a, b), 10)) / 2) for _ in range(2)],
    # 'x||dist_log': [(lambda a, b: math.log(KL(a, b), 10)) for _ in range(2)],
    'dist||x_log': [(lambda a, b: math.log(KL(b, a), 10)) for _ in range(2)],
    # 'dist||x_clean_log': [(lambda a, b: math.log(KL(b, a), 10)), lambda a, b: 0],
    # 'x||dist_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(a, b), 10))],
    # 'dist||x_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))]
}

temp = None

for op_code_type in ['common_cluster']:

    results = {}
    for test_key, test_path in {
        # 'TRAIN': MALICIOUS_TRAIN_PATH,
        # 'VS_000': VIRUS_SHARE_000,
        # 'VS_005': VIRUS_SHARE_005,
        'VS_451': VIRUS_SHARE_451,
        # 'MAC_OS': MAC_OS_PATH,
        # 'WIN_OS': '/Volumes/T7/Windows/'
    }.items():
        for seed in [1, 9, 83, 85]:
            for i in range(10):

                data = get_distribution_data(
                    op_code_directory=test_path,
                    path=DISTRIBUTION_PATH,
                    iteration=i,
                    distribution_funcs=distributions,
                    sample_size=SAMPLE_SIZE,
                    bins=BINS,
                    op_code_types=[op_code_type],
                    training_size=500,
                    kl_funcs=kls,
                    method=METHOD,
                    pruned=False,
                    random_seed=SEEDS[i] * seed,
                    update_text_prefix=f"{test_key}/{seed}/{i} - ",
                    exclude_files=False
                )

                # TODO - run for all KLs and models
                # TODO - create giant heat map plot, with training data
                for training_method in ['mlp_scaled']:
                    for d in data:
                        for b in BINS:
                            for kl_func in data[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs']:

                                f = f"{op_code_type}_{training_method}_{b}_{kl_func}_{i}"
                                k = test_key + '_' + f"{op_code_type}_{training_method}_{b}_{kl_func}"
                                if k not in results:
                                    results.update({k: []})
                                loaded_model = pickle.load(
                                    open(f'./models/{f}.pickle', "rb")
                                )
                                # print(loaded_model.coefs_)
                                # print('\n\n')
                                # print(data[d]['op_code_types'][OP_CODE_TYPE]['bins'][b]['kl_funcs'][kl_func]['X'].shape)
                                # print(data[d]['op_code_types'][OP_CODE_TYPE]['bins'][b]['kl_funcs'][kl_func]['y'].shape)
                                X_test = data[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X']
                                y_test = data[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['y'].reshape(-1)
                                # X_test = np.load(f'./Example_{SAMPLE_SIZE}.npy')
                                # y_test = np.zeros(SAMPLE_SIZE)

                                ss = X_test.shape[0]
                                # if temp is None:
                                #     temp = X_test[0]
                                # else:
                                #     print(temp == X_test[0])

                                y_test = y_test.astype(int)
                                # print(y_test)
                                prediction = loaded_model.predict(X_test)

                                # print(f, '-', )
                                results[k] += [sum(prediction == y_test) / ss]

    for f in results:
        mean = sum(results[f]) / len(results[f])
        var = ((sum([((z - mean) ** 2) for z in results[f]]) / len(results[f])) ** .5) * 100
        print(f"{f}, mean: {mean * 100:.3f}, var: {var:.3f}")
        print(results[f])

from KL_models import get_distribution_data, KL

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

SAMPLE_SIZE = 1470
METHOD = "jump"
OP_CODE_TYPE = "common_cluster"
MALICIOUS_TEST_PATH = "/Volumes/T7/VirusShare/"
DISTRIBUTION_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples" \
                    f"/{METHOD}/base/{OP_CODE_TYPE}/op_codes"

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
    # 'dist||x_infected_log': [lambda a, b: 0, (lambda a, b: math.log(KL(b, a), 10))]
}

temp = None

results = {}
for i in range(1):

    data = get_distribution_data(
        op_code_directory=MALICIOUS_TEST_PATH,
        path=DISTRIBUTION_PATH,
        iteration=5,  # random.randint(0, 9),
        distribution_funcs=distributions,
        sample_size=SAMPLE_SIZE,
        bins=[100],
        op_code_type=OP_CODE_TYPE,
        training_size=500,
        kl_funcs=kls,
        method=METHOD,
        pruned=False,
        random_seed=SEEDS[i],
        update_text_prefix=f"{i} - "
    )

    # TODO - run for all KLs and models
    # TODO - create giant heat map plot, with training data
    for training_method in ['mlp']:
        for d in data:
            for b in data[d]['bins']:
                for kl_func in data[d]['bins'][b]['kl_funcs']:
                    f = f"{training_method}_{b}_{kl_func}"
                    if f not in results:
                        results.update({f: []})
                    loaded_model = pickle.load(
                        open(f'./models/{f}.pickle', "rb")
                    )
                    # print(loaded_model.coefs_)

                    X_test = data[d]['bins'][b]['kl_funcs'][kl_func]['X']
                    y_test = data[d]['bins'][b]['kl_funcs'][kl_func]['y'].reshape(-1)
                    # X_test = np.load(f'./Example_{SAMPLE_SIZE}.npy')
                    # y_test = np.zeros(SAMPLE_SIZE)

                    ss = X_test.shape[0]
                    # if temp is None:
                    #     temp = X_test[0]
                    # else:
                    #     print(temp == X_test[0])

                    y_test = y_test.astype(int)

                    prediction = loaded_model.predict(X_test)
                    # print(prediction)
                    # print(f, '-', )
                    results[f] += [sum(prediction == y_test) / ss]

for f in results:
    print(f, results[f])

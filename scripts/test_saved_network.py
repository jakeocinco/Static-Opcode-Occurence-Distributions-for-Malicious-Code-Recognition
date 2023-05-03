from train_models import get_distribution_data, KL
from base_functions import *

import pickle

SEEDS = [198, 598, 430, 206, 602, 992, 786, 527, 23, 371, 175, 291, 960, 706, 736, 750, 748, 247, 664, 999, 559, 570,
         584, 727, 257]

distributions = {
    'linear': lambda a, b: a / 1000,
}

SAMPLE_SIZE = 50
BINS = [100]
METHOD = "jump"

kls = ['dist||x_log']
models = ['mlp_scaled']
temp = None

for op_code_type in ['infected']:

    results = {}
    for test_key in ['VirusShare00005']:
        for seed in [1, 9]:
            for i in range(2):

                data = get_distribution_data(
                    iteration=i,
                    distribution_funcs=distributions,
                    sample_size=SAMPLE_SIZE,
                    bins=BINS,
                    op_code_types=[op_code_type],
                    cumulative_distribution_samples_used=500,
                    kl_funcs={kl: KL_METHODS[kl] for kl in kls},
                    method=METHOD,
                    pruned=False,
                    random_seed=SEEDS[i] * seed,
                    update_text_prefix=f"{test_key}/{seed}/{i} - ",
                    exclude_files=False,
                    sample_names=[test_key]
                )

                for training_method in models:
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

                                X_test = data[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['X']
                                y_test = data[d]['op_code_types'][op_code_type]['bins'][b]['kl_funcs'][kl_func]['y'].reshape(-1)

                                number_of_samples = X_test.shape[0]

                                y_test = y_test.astype(int)
                                prediction = loaded_model.predict(X_test)

                                results[k] += [sum(prediction == y_test) / number_of_samples]

    for f in results:
        mean = sum(results[f]) / len(results[f])
        var = ((sum([((z - mean) ** 2) for z in results[f]]) / len(results[f])) ** .5) * 100
        print(f"{f}, mean: {mean * 100:.3f}, var: {var:.3f}")
        print(results[f])

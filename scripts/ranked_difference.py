import json
import scipy.stats
import sys

KL_CANDIDATES = ['two_sided', 'x||dist', 'dist||x', 'x||dist_clean', 'dist||x_clean', 'x||dist_infected',
                 'dist||x_infected', 'two_sided_log', 'x||dist_log', 'dist||x_log', 'x||dist_clean_log',
                 'dist||x_clean_log', 'x||dist_infected_log', 'dist||x_infected_log']
MODEL_CANDIDATES = ['linear_svm', 'linear_svm_scaled', 'ridge', 'ridge_scaled', 'sgd', 'logistic', 'logistic_scaled',
                    'mlp', 'mlp_scaled']
OP_CANDIDATES = ['benign', 'infected', 'union', 'intersection', 'disjoint', 'ratio', 'ratio_a25', 'ratio_a75',
                   'malware_cluster', 'common_cluster', 'infected_common_cluster', 'infected_ratio_a75',
                 'infected_ratio_a75_common_cluster', 'jump_share', 'ratio_a75_common_cluster']
FILES = ['jump_9_1680727145.json']
METHOD = 'jump'
OP_CODES = ['benign', 'infected', 'union', 'intersection', 'disjoint']
ITERATIONS = range(10)
BINS = [100]
KL = ['x||dist']
MODELS = ['mlp']


def print_res(text, _long, Z):
    p = scipy.stats.norm.sf(Z)
    print(f"{text} {' ' * (_long - len(text))}| Z:{Z:.2f} - p:{1 - p:.3f}")


def _rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)


def _ranked_sign_test(diff):

    l = len(diff)
    sign = [0 if x < 0 else 1 for x in diff]
    rank = list(scipy.stats.rankdata([abs(x) for x in diff]))
    # print(sign)
    # print(rank)
    # rank = _rank_simple([abs(x) for x in diff])
    # print(rank)
    expected_vs = (l * (l + 1)) / 4
    var_vs = (l * (l + 1) * ((2 * l) + 1)) / 24
    vs = sum([sign[i] * rank[i] for i in range(l)])

    # print(vs)
    # print(expected_vs, var_vs, vs)

    if vs == 0:
        vs = 1
    z_score = (vs + 0.5 - expected_vs) / (var_vs ** .5)

    return z_score


def get_results(
        seeds,
        method,
        op_codes,
        iterations,
        bins,
        kls,
        models,
        pruned=False
):
    data = []
    pruned_path = "pruned" if pruned else "base"
    for seed in seeds:

        results = json.load(
            open(f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/{method}_{seed}.json", "r")
        )

        for iteration in results[pruned_path]:
            if int(iteration) in iterations:
                for op in results[pruned_path][iteration]['linear']:
                    if op in op_codes:
                        for bin in results[pruned_path][iteration]['linear'][op]:
                            if int(bin) in bins:
                                for kl in results[pruned_path][iteration]['linear'][op][bin]:
                                    if kl in kls:
                                        for m in results[pruned_path][iteration]['linear'][op][bin][kl]:
                                            if m in models:
                                                data += [results[pruned_path][iteration]['linear'][op][bin][kl][m]]

    return data


def pairwise_op(
        seeds,
        method,
        iterations,
        bins,
        kls,
        models
):
    results = []
    diff = []
    for _op in ['benign', 'infected']:
        results += [get_results(seeds, method, [_op], iterations, bins, kls, models)]
    for r in range(len(results[0])):
        diff += [results[1][r] - results[0][r]]

    # print([f"{x:.3f}" for x in results[0]])
    # print([f"{x:.3f}" for x in results[1]])

    Z = _ranked_sign_test(diff)

    p = scipy.stats.norm.sf(Z)

    print(f"benign vs malicious | Z:{Z:.2f} - p:{1 - p:.3f} (p malicious better than benign)")

    _long = max([len(x) for x in OP_CANDIDATES])
    for comp_op in ['benign', 'infected', 'union']:
        print(f'\n{comp_op.capitalize()} vs other')
        for op in OP_CANDIDATES:
            if comp_op != op:
                results = []
                diff = []
                for _op in [comp_op, op]:
                    results += [get_results(seeds, method, [_op], iterations, bins, kls, models)]
                for r in range(len(results[0])):
                    diff += [results[1][r] - results[0][r]]

                if op == 'jump_share':

                    print([f"{x:.3f}" for x in results[0]])
                    print([f"{x:.3f}" for x in results[1]])
                    print([f"{x:.3f}" for x in diff])

                Z = _ranked_sign_test(diff)
                print_res(op, _long, Z)


def pairwise_kl(
        seeds,
        method,
        op_codes,
        iterations,
        bins,
        models
):
    for _KL in ['x||dist', 'dist||x', 'two_sided']:
        print(f'KL methods against {_KL}')
        kls = ['x||dist', 'dist||x', 'two_sided']
        if _KL != 'two_sided':
            kls += [f'{_KL}_clean',  f'{_KL}_infected']

        _long = max([len(x) for x in kls])
        for kl in kls:
            if kl != _KL:
                results = []
                diff = []
                for _kl in [_KL, kl]:
                    results += [get_results(seeds, method, op_codes, iterations, bins, [_kl], models)]
                for r in range(len(results[0])):
                    diff += [results[1][r] - results[0][r]]

                # print([f"{x:.3f}" for x in results[0]])
                # print([f"{x:.3f}" for x in results[1]])
                # print([f"{x:.3f}" for x in diff])

                Z = _ranked_sign_test(diff)

                # print(f"{kl} | Z:{Z:.2f} - p:{1 - p:.3f}")
                print_res(kl, _long, Z)
        print()

    full_diff = []
    print('Logarithm Comparisons')
    kl_options = [
        ('x||dist', 'x||dist_log'),
        ('dist||x', 'dist||x_log'),
        ('two_sided', 'two_sided_log'),
        # ('x||dist_clean', 'x||dist_clean_log'),
        # ('dist||x_clean', 'dist||x_clean_log'),
        # ('x||dist_infected', 'x||dist_infected_log'),
        # ('dist||x_infected', 'dist||x_infected_log')
    ]
    _long = max([(len(x[0]) + len(x[1]) + 3) for x in kl_options])
    for kls in kl_options:
        results = []
        diff = []
        for _kl in kls:
            results += [get_results(seeds, method, op_codes, iterations, bins, [_kl], models)]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]
        full_diff += diff

        Z = _ranked_sign_test(diff)
        print_res(f"{kls[0]} - {kls[1]}", _long, Z)

    Z = _ranked_sign_test(full_diff)
    print_res("all methods log", _long, Z)

    for kl_option in ['x||dist', 'two_sided_log', 'x||dist_log', 'dist||x_log']:
        print(f'\nKL methods against {kl_option}')
        log_kls = ['x||dist',  'dist||x',  'two_sided', 'two_sided_log', 'x||dist_log', 'dist||x_log']
        _long = max([len(x) for x in log_kls])
        for kl in log_kls:
            if kl != kl_option:
                results = []
                diff = []
                for _kl in [kl_option, kl]:
                    results += [get_results(seeds, method, op_codes, iterations, bins, [_kl], models)]
                for r in range(len(results[0])):
                    diff += [results[1][r] - results[0][r]]

                # print([f"{x:.3f}" for x in results[0]])
                # print([f"{x:.3f}" for x in results[1]])
                # print([f"{x:.3f}" for x in diff])

                Z = _ranked_sign_test(diff)
                print_res(kl, _long, Z)


def pairwise_models(
        seeds,
        method,
        op_codes,
        iterations,
        bins,
        kls
):
    print('Compared to linear svm')
    models = ['ridge', 'sgd', 'logistic', 'mlp']
    _long = max([len(x) for x in models + ['linear_svm']])
    for model in models:
        results = []
        diff = []
        for _model in ['linear_svm', model]:
            results += [get_results(seeds, method, op_codes, iterations, bins, kls, [_model])]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]

        Z = _ranked_sign_test(diff)
        print_res(model, _long, Z)

    print('\nCompared to mlp')
    for model in ['linear_svm', 'ridge', 'sgd', 'logistic']:
        results = []
        diff = []
        for _model in ['mlp', model]:
            results += [get_results(seeds, method, op_codes, iterations, bins, kls, [_model])]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]

        Z = _ranked_sign_test(diff)
        print_res(model, _long, Z)

    full_diff = []
    print('\nScaled Comparisons')
    model_options = [
        ('linear_svm', 'linear_svm_scaled'),
        ('ridge', 'ridge_scaled'),
        ('logistic', 'logistic_scaled'),
        ('mlp', 'mlp_scaled')
    ]
    _long = max([(len(x[0]) + len(x[1]) + 3) for x in model_options])
    for models in model_options:
        results = []
        diff = []
        for _model in models:
            results += [get_results(seeds, method, op_codes, iterations, bins, kls, [_model])]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]

        full_diff += diff
        Z = _ranked_sign_test(diff)
        print_res(f"{models[0]} - {models[1]}", _long, Z)

    Z = _ranked_sign_test(full_diff)
    p = scipy.stats.norm.sf(Z)

    print_res(f"all models scaled", _long, Z)

    print('\nCompared to mlp_scaled')
    for model in ['linear_svm_scaled', 'ridge_scaled', 'sgd', 'logistic_scaled']:
        results = []
        diff = []
        for _model in ['mlp_scaled', model]:
            results += [get_results(seeds, method, op_codes, iterations, bins, kls, [_model])]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]

        Z = _ranked_sign_test(diff)
        p = scipy.stats.norm.sf(Z)

        print(f"{model} | Z:{Z:.2f} - p:{1 - p:.3f}")


def pairwise_pruned(
        seeds,
        method,
        iterations,
        bins,
        kls,
        models
):
    print('Pruned data')
    diffs = []
    for op in ['benign', 'infected', 'union', 'intersection', 'disjoint']:
        results = []
        diff = []
        for _pruned in [False, True]:
            results += [get_results(seeds, method, [op], iterations, bins, kls, models, _pruned)]
        for r in range(len(results[0])):
            diff += [results[1][r] - results[0][r]]

        diffs += diff

        # print('base', results[0])
        # print('pruned', results[1])
        # print(diff)
        Z = _ranked_sign_test(diff)
        p = scipy.stats.norm.sf(Z)

        print(f"{op} | Z:{Z:.2f} - p:{1 - p:.3f}")

    Z = _ranked_sign_test(diffs)
    p = scipy.stats.norm.sf(Z)

    print(f"all samples | Z:{Z:.2f} - p:{1 - p:.3f}")


def pairwise_bins(
        seeds,
        method,
        ops,
        iterations,
        kls,
        models
):
    print('Bins')
    results = []
    diff = []
    for _bin in [25, 100]:
        results += [get_results(seeds, method, ops, iterations, [25], kls, models)]
    for r in range(len(results[0])):
        diff += [results[1][r] - results[0][r]]

    Z = _ranked_sign_test(diff)
    p = scipy.stats.norm.sf(Z)

    print(f"25 v 100 | Z:{Z:.2f} - p:{1 - p:.3f}")


def pairwise_method(
        seeds,
        methods,
        op_codes,
        iterations,
        bins,
        kls,
        models
):
    for _method in ['jump', 'share']:
        for method in ['share', 'cumulative_share']:
            if _method != method:
                for op in op_codes:
                    results = []
                    diff = []
                    for m in [_method, method]:
                        results += [get_results(seeds, m, [op], iterations, bins, kls, models)]
                    for r in range(len(results[0])):
                        diff += [results[1][r] - results[0][r]]

                    # print([f"{x:.3f}" for x in results[0]])
                    # print([f"{x:.3f}" for x in results[1]])

                    Z = _ranked_sign_test(diff)

                    p = scipy.stats.norm.sf(Z)

                    print(f"{_method} vs {method}  - {op}| Z:{Z:.2f} - p:{1 - p:.3f} (p {method} better than {_method})")
                print()


if __name__ == "__main__":

    SEEDS = [1, 9, 83, 85]
    count = 0
    if 'op' in sys.argv:
        print(' -- Op Code Sets --')
        pairwise_op(
            seeds=SEEDS,
            method='jump',
            iterations=ITERATIONS,
            bins=BINS,
            kls=['dist||x_log'],
            models=['mlp_scaled']
        )
        count += 1

    if 'kl' in sys.argv:
        if count > 0:
            print()
        print(' -- KL Methods --')
        pairwise_kl(
            seeds=SEEDS,
            method='jump',
            op_codes=['infected'],
            iterations=ITERATIONS,
            bins=BINS,
            models=['mlp_scaled']
        )
        count += 1

    if 'model' in sys.argv:
        if count > 0:
            print()
        print(' -- Models --')
        pairwise_models(
            seeds=SEEDS,
            method='jump',
            op_codes=OP_CODES,
            iterations=ITERATIONS,
            bins=BINS,
            kls=KL_CANDIDATES
        )
        count += 1

    if 'bins' in sys.argv:
        if count > 0:
            print()
        print(' -- Bins --')
        pairwise_bins(
            seeds=SEEDS,
            method='jump',
            iterations=ITERATIONS,
            ops=['benign'],
            kls=['x||dist'],
            models=['mlp_scaled']
            # kls=['x||dist'],
            # models=['mlp']
        )
        count += 1

    if 'prune' in sys.argv:
        if count > 0:
            print()
        print(' -- Pruned --')
        pairwise_pruned(
            seeds=SEEDS,
            method='jump',
            iterations=ITERATIONS,
            bins=BINS,
            # kls=KL_CANDIDATES,
            # models=MODEL_CANDIDATES
            kls=['x||dist'],
            models=['mlp_scaled']
        )
        count += 1

    if 'method' in sys.argv:
        if count > 0:
            print()
        print(' -- Method --')
        pairwise_method(
            seeds=SEEDS,
            methods=['jump', 'share'],
            op_codes=['benign', 'infected', 'union', 'intersection', 'disjoint'],
            iterations=ITERATIONS,
            bins=BINS,
            # kls=KL_CANDIDATES,
            # models=MODEL_CANDIDATES
            kls=['dist||x_log'],
            models=['mlp_scaled']
        )
        count += 1
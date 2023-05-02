import os
import json

RESULTS_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/"

def get_most_recent_result(f):

    recent_results = {}
    results = os.listdir(RESULTS_PATH)
    f_pruned = 'base' if not f['pruned'] else 'pruned'
    prefix = f"{f['method']}_{f_pruned}_"
    f_results = list(
        filter(
            lambda x: prefix == x[:len(prefix)],
            results
        )
    )

    for r in f_results:
        with open(RESULTS_PATH + r, 'r') as f:
            file = f.read()
        data = json.loads(file)
        recent_results.update(data)

    return recent_results


def compare_most_recent(
        f1, f2
):

    results_f1 = get_most_recent_result(f1)
    results_f2 = get_most_recent_result(f2)

    keys = list(set(list(results_f1.keys()) + list(results_f2.keys())))

    f1_arr = {}
    f2_arr = {}
    for key in keys:
        arr = []
        print(key)
        r1 = results_f1[key]['5']['linear']
        r2 = results_f2[key]['5']['linear']
        for b in ['25', '100']:
            for k2 in r1[b].keys():
                if k2 in r2[b]:
                    for learning_method in r1[b][k2].keys():
                        if learning_method in r2[b][k2]:
                            arr += [r1[b][k2][learning_method] - r2[b][k2][learning_method]]
                            f1_arr.update({
                                r1[b][k2][learning_method]: f"{key}, {b}, {k2}, {learning_method}"
                            })
                            # f1_arr += [r1[b][k2][learning_method]]
                            f2_arr.update({
                                r2[b][k2][learning_method]: f"{key}, {b}, {k2}, {learning_method}"
                            })

        print(sum(arr) / len(arr))
        print(max(arr), min(arr))
        # print(sum(f1_arr.keys()) / len(f1_arr.keys()), max(f1_arr.keys()))
        # print(sum(f2_arr) / len(f2_arr), max(f2_arr))

    print(1)
    print({i: f1_arr[i] for i in sorted(f1_arr.keys())})
    print(2)
    print({i: f2_arr[i] for i in sorted(f2_arr.keys())})
f1 = {
    'method': 'share',
    'pruned': False,
    'iteration': 5
}

f2 = {
    'method': 'inverse_jump',
    'pruned': False,
    'iteration': 5
}

compare_most_recent(f1, f2)
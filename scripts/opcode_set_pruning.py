import numpy as np
from sklearn.svm import SVC

from base_functions import  *

# This script prunes opcodes from opcode sets by using Linear SVM to determine whether the opcodes are
# separable by classes

def _create_all_possible_sub_directories(base_path):

    directories = find_all_sub_paths(base_path)
    # directories = list(set([item for sublist in directories for item in sublist]))
    directories.sort(key=len)
    for directory in directories:
        # print('-', directory)
        make_directory(directory)


def select_pruned_ops(
    op_code_distribution_path,
    distribution,
    bins,
    sample_size,
    method,
    op_code_sample
):

    base_path = f"{op_code_distribution_path}/{method}/base/{op_code_sample}/op_codes/{distribution}/" \
                 f"{bins}_bins/{sample_size}_samples"

    try:
        clean_arr = os.listdir(f"{base_path}/clean")
        infected_arr = os.listdir(f"{base_path}/infected")

        if '.DS_Store' in clean_arr:
            clean_arr.remove('.DS_Store')
        if '.DS_Store' in infected_arr:
            infected_arr.remove('.DS_Store')

        clean_arr = list(
            map(
                lambda x: np.load(f"{base_path}/clean/{x}"),
                filter(
                    lambda x: "._" not in x and '.npy' in x,
                    clean_arr
                )
            )
        )
        infected_arr = list(
            map(
                lambda x: np.load(f"{base_path}/infected/{x}"),
                filter(
                    lambda x: "._" not in x and '.npy' in x,
                    infected_arr
                )
            )
        )
        sample_image_data = clean_arr[0]

        combined = clean_arr + infected_arr
        combined_length = len(combined)

        data = np.zeros((combined_length, combined_length))

        index = 0

        x = {}

        # data = np.zeros((sample_image_data.shape[0], combined_length))
        # print(data.shape)
        clean_len = len(clean_arr)
        y = np.array([1 for _ in range(len(clean_arr))] + [0 for _ in range(len(infected_arr))])

        print(op_code_sample, y)
        true_ops = []
        for k in range(sample_image_data.shape[0]):
            for i, id in enumerate(combined):
                for j, jd in enumerate(combined):
                    data[i, j] = KL(id[k], jd[k])
            svc = SVC(kernel='precomputed')
            svc.fit(data, y)

            pred = svc.predict(data)
            truth = (sum(pred[:clean_len]) == 10 or sum(pred[:clean_len]) == 0) and\
                    (sum(pred[clean_len:]) == 10 or sum(pred[clean_len:]) == 0)
            # print(OP_CODE_DICT[op_code_sample][k], truth)
            # print(pred)
            if truth:
                true_ops += [OP_CODE_DICT['base'][op_code_sample][k]]
        return true_ops
    except: print(op_code_sample, [])


def create_pruned_distribution_sets(
    distribution,
    bins,
    sample_size,
    method,
    op_code_sample
):
    base_path = f"{DISTRIBUTION_SAMPLE_PATH}/{method}/base/{op_code_sample}/op_codes/{distribution}/" \
                f"{bins}_bins/{sample_size}_samples"
    prune_path = f"{DISTRIBUTION_SAMPLE_PATH}/{method}/pruned/{op_code_sample}/op_codes/{distribution}/" \
                f"{bins}_bins/{sample_size}_samples"

    clean_arr = os.listdir(f"{base_path}/clean")
    infected_arr = os.listdir(f"{base_path}/infected")

    if '.DS_Store' in clean_arr:
        clean_arr.remove('.DS_Store')
    if '.DS_Store' in infected_arr:
        infected_arr.remove('.DS_Store')

    clean_arr = list(
        map(
            lambda x: f"/clean/{x}",
            filter(
                lambda x: "._" not in x and '.npy' in x,
                clean_arr
            )
        )
    )
    infected_arr = list(
        map(
            lambda x: f"/infected/{x}",
            filter(
                lambda x: "._" not in x and '.npy' in x,
                infected_arr
            )
        )
    )
    sample_image_data = np.load(f"{base_path}/{clean_arr[0]}")

    combined = clean_arr + infected_arr

    for i, id in enumerate(combined):
        base_image = np.load(f"{base_path}/{id}")
        data = []

        for k in range(sample_image_data.shape[0]):
            if OP_CODE_DICT['base'][op_code_sample][k] in OP_CODE_DICT[method][op_code_sample]:
                data += [base_image[k]]
                if method == 'jump':
                    if sum(base_image[k]) == 0:
                        base_image[k] += 1
                    base_image[k] *= 1 / sum(base_image[k])

        data = np.array(data)
        _create_all_possible_sub_directories(prune_path + '/infected/')
        _create_all_possible_sub_directories(prune_path + '/clean/')

        if method in ['inverse_jump', 'share', 'cumulative_share']:
            for k in range(sample_image_data.shape[1]):
                if sum(data[:, k]) == 0:
                    data[:, k] += 1
                data[:, k] = 1 / sum(data[:, k])

        with open(f"{prune_path}/{id}", 'wb') as file:
            np.save(file, data)


if __name__ == "__main__":

    op_code_samples = ['infected']

    for method in ['jump']:

        for op_code_sample in op_code_samples:
            print(op_code_sample, method)
            for b in [25, 100]:
                create_pruned_distribution_sets(
                    distribution='linear',
                    bins=b,
                    sample_size=500,
                    method=method,
                    op_code_sample=op_code_sample
                )

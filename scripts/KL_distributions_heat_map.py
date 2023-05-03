import numpy as np
import matplotlib.pyplot as plt

from base_functions import  *

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(a * np.log(a / b), 0)


def get_difference_matrix(
    op_code_distribution_path,
    distribution,
    bins,
    sample_size,
    method,
    rows=None
):
    clean_path = f"{op_code_distribution_path}/{distribution}/{bins}_bins/{sample_size}_samples/clean"
    infected_path = f"{op_code_distribution_path}/{distribution}/{bins}_bins/{sample_size}_samples/infected"

    clean_arr = os.listdir(clean_path)
    infected_arr = os.listdir(infected_path)

    if '.DS_Store' in clean_arr:
        clean_arr.remove('.DS_Store')
    if '.DS_Store' in infected_arr:
        infected_arr.remove('.DS_Store')

    clean_arr = list(
        filter(
            lambda x: "._" not in x and '.npy' in x,
            clean_arr
        )
    )
    infected_arr = list(
        filter(
            lambda x: "._" not in x and '.npy' in x,
            infected_arr
        )
    )

    sample_image_data = np.load(f"{clean_path}/{clean_arr[0]}")
    print(sample_image_data.shape)
    combined_length = len(clean_arr) + len(infected_arr)
    data = np.zeros((combined_length, sample_image_data.shape[0], sample_image_data.shape[1]))

    _rows = rows if rows is not None else range(sample_image_data.shape[0])
    index = 0
    for temporary_data in [
        {
            'arr': clean_arr,
            'path': clean_path
        },
        {
            'arr': infected_arr,
            'path': infected_path
        }
    ]:
        for image_path in temporary_data['arr']:
            if "._" not in image_path and ".npy" in image_path:

                image_data = np.load(f"{temporary_data['path']}/{image_path}")
                # print(image_data)
                for row in _rows:
                    # if sorted(list(OP_CODE_CLUSTER['malware_cluster'].values())) == 'mov':
                    image_data[row, :] += 1.0
                    image_data[row, :] *= 1.0 / sum(image_data[row, :])

                data[index, :, :] = image_data[:, :]
                index += 1

    differences = np.zeros((combined_length, combined_length))
    for i in range(combined_length):
        for j in range(combined_length):
            if method in ['jump']:
                for row in _rows:
                    value = KL(data[i, row], data[j, row])
                    differences[i, j] += value
            elif method in ['share', 'cumulative_share', 'inverse_jump']:
                for row in range(sample_image_data.shape[1]):
                    value = KL(data[i, :, row], data[j, :, row])
                    differences[i, j] += value
    return {
        'differences': differences,
        'split_index': len(clean_arr)
    }


def distributions_heat_map(
    distribution,
    bins,
    sample_size,
    op_sample,
    method,
    pruned=False
):
    prune_path = 'base' if not pruned else 'pruned'
    _op_code_distribution_path = f"{DISTRIBUTION_SAMPLE_PATH}/{METHOD}/{prune_path}/{op_sample}/op_codes"

    differences = get_difference_matrix(
        op_code_distribution_path=_op_code_distribution_path,
        distribution=distribution,
        bins=bins,
        sample_size=sample_size,
        method=method
    )['differences']

    plt.imshow(differences, cmap='hot', interpolation=None)
    plt.title(f"{op_sample.capitalize()} - bins: {bins}, samples: {sample_size}")
    plt.show()


if __name__ == "__main__":
    METHOD = "jump"
    op_code_samples = ['benign', 'infected', 'union', 'intersection', 'disjoint']
    for op_code_sample in op_code_samples:
        for distributions in ['linear']: #, 'log10', 'log100', 'threshold']:
            for bins in [100]:
                distributions_heat_map(
                    distribution=distributions,
                    bins=bins,
                    sample_size=500,
                    op_sample=op_code_sample,
                    method=METHOD,
                    pruned=False
                )

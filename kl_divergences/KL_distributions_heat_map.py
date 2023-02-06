import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(a * np.log(a / b), 0)


def get_difference_matrix(
    op_code_distribution_path,
    distribution,
    bins,
    sample_size,
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

    combined_length = len(clean_arr) + len(infected_arr)
    data = np.zeros((combined_length, sample_image_data.shape[0], sample_image_data.shape[1]))

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
                for row in range(sample_image_data.shape[0]):
                    image_data[row, :] += 1.0
                    image_data[row, :] *= 1.0 / sum(image_data[row, :])

                data[index, :, :] = image_data[:, :]
                index += 1

    differences = np.zeros((combined_length, combined_length))
    for i in range(combined_length):
        for j in range(combined_length):
            for row in range(sample_image_data.shape[0]):
                value = KL(data[i, row], data[j, row])
                # print(value)
                differences[i, j] += value

    return {
        'differences': differences / differences.max(),
        'split_index': len(clean_arr)
    }


def distributions_heat_map(
    op_code_distribution_path,
    distribution,
    bins,
    sample_size,
):

    differences = get_difference_matrix(
        op_code_distribution_path=op_code_distribution_path,
        distribution=distribution,
        bins=bins,
        sample_size=sample_size
    )['differences']

    plt.imshow(differences, cmap='hot', interpolation=None)
    plt.title(f"{distribution}, bins: {bins}, samples: {sample_size}")
    plt.show()


def distribution_efficacy(
    op_code_distribution_path,
    distributions,
    bins,
    op_codes,
    sample_size,
):
    for op_code in op_codes:
        for distribution in distributions:
            for b in bins:
                distrib = get_difference_matrix(
                    op_code_distribution_path=op_code_distribution_path + f"{op_code}/op_codes",
                    distribution=distribution,
                    bins=b,
                    sample_size=sample_size
                )
                differences = distrib['differences']
                split_index = distrib['split_index']

                total = 0

                for r in range(differences.shape[0]):
                    for c in range(differences.shape[1]):
                        if (r < split_index and c < split_index) or (r >= split_index and c >= split_index):
                            total -= differences[r, c]
                        else:
                            total += differences[r, c]

                total /= 2 * ((differences.shape[0] - split_index) ** 2)

                print(f"{op_code.capitalize()}, Distribution: {distribution}, Bins: {b} - {total}")
        print()


if __name__ == "__main__":
    op_code_samples = ['ratio_a75']
    for op_code_sample in op_code_samples:
        for distributions in ['linear', 'log10']: #, 'log100', 'threshold']:
            for bins in [25, 100]:
                distributions_heat_map(
                    op_code_distribution_path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/"
                                              f"op_code_distributions_samples/{op_code_sample}/op_codes",
                    distribution=distributions,
                    bins=bins,
                    sample_size=500
                )

    # distributions = ['ration'] # , 'log100', 'threshold'
    # bins = [25, 100]
    # distribution_efficacy(
    #     op_code_distribution_path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/"
    #                               f"op_code_distributions_samples/",
    #     distributions=distributions,
    #     bins=bins,
    #     op_codes=op_code_samples,
    #     sample_size=500
    # )
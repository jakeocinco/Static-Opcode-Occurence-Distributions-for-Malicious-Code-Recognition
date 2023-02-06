import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import os


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(a * np.log(a / b), 0)


def distributions_heat_map(
    op_code_distribution_path,
    distribution,
    bins,
    sample_size,
    op_codes
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
            lambda x: "._" not in x,
            clean_arr
        )
    )
    infected_arr = list(
        filter(
            lambda x: "._" not in x,
            infected_arr
        )
    )

    sample_image = Image.open(f"{clean_path}/{clean_arr[0]}")
    sample_image_data = np.asarray(sample_image)
    sample_image_data = sample_image_data.astype(np.float)

    combined_length = len(clean_arr) + len(infected_arr)
    data = np.zeros((combined_length, len(op_codes), sample_image_data.shape[1]))

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
            if "._" not in image_path:
                image = Image.open(f"{temporary_data['path']}/{image_path}")
                image_data = np.asarray(image).astype(np.float)
                for row in range(sample_image_data.shape[0]):

                    image_data[row, :] += 1.0
                    image_data[row, :] *= 1.0 / sum(image_data[row, :])

                data[index, :, :] = image_data[:, :, 0]
                index += 1

    for index, op in enumerate(op_codes):
        differences = np.zeros((combined_length, combined_length))
        for i in range(combined_length):
            for j in range(combined_length):
                value = KL(data[i, index], data[j, index]) + KL(data[j, index], data[i, index])
                differences[i, j] = (value / 2)

        model = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
        model.fit(differences)
        print(op, model.labels_)


if __name__ == "__main__":

    independent_OP_CODES =  ['andb', 'andw', 'divl', 'faddl', 'flds', 'insl', 'jbe', 'lock', 'lock', 'outsl', 'rorb', 'shrb']

    for distributions in ['linear', 'log10', 'log100', 'threshold']:
        for bins in [25, 100]:
            print(bins, distributions)
            distributions_heat_map(
                op_code_distribution_path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples"
                                          f"/dependent/op_codes",
                distribution=distributions,
                bins=bins,
                sample_size=500,
                op_codes=independent_OP_CODES
            )
            print("")

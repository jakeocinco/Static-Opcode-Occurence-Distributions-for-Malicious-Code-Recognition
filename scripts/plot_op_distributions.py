import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os


def plot_op_distributions(
        op_code_distribution_path,
        distribution,
        bins,
        sample_size,
        iteration,
        shape,
        op_codes=None
):

    base = f"{op_code_distribution_path}/{distribution}/{bins}_bins/{sample_size}_samples"

    clean_image = Image.open(f"{base}/clean/{iteration}.jpeg")
    infected_image = Image.open(f"{base}/infected/{iteration}.jpeg")

    clean_image_data = np.asarray(clean_image).astype(np.float)
    infected_image_data = np.asarray(infected_image).astype(np.float)

    # fig, axs = plt.subplots(shape[0], shape[1])
    fig, axs = plt.subplots(2, clean_image_data.shape[0])
    fig.set_size_inches(16, 6)
    fig.suptitle(f"Bins: {bins}, Distribution: {distribution}")
    for r in range(clean_image_data.shape[0]):
        a = np.asarray([x + 1 for x in range(bins)])
        # axs[int(r // shape[1]), int(r % shape[1])].hist(
        axs[0, r].hist(
            a,
            clean_image_data.shape[1],
            alpha=0.5,
            weights=clean_image_data[r, :, 0]
        )
        # axs[int(r // shape[1]), int(r % shape[1])].hist(
        axs[1, r].hist(
            a,
            infected_image_data.shape[1],
            alpha=0.5,
            weights=infected_image_data[r, :, 0]
        )

        if op_codes is not None:
            axs[1, r].set_title(op_codes[r])
            # axs[int(r // shape[1]), int(r % shape[1])].set_title(op_codes[r])

    plt.show()


if __name__ == "__main__":

    for distributions in ['linear', 'log10', 'log100', 'threshold']:
        for bins in [25, 100]:
            plot_op_distributions(
                op_code_distribution_path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples"
                                          f"/independent_diff/op_codes",
                distribution=distributions,
                bins=bins,
                sample_size=500,
                iteration=1,
                shape=(2, 3),
                op_codes=['aas', 'fstps', 'jb', 'je', 'movsb', 'rorl', 'sbbl', 'sete', 'testw']
            )
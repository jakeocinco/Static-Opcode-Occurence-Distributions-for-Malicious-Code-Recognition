import os
import json

from op_codes import *
import math
import random

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# def create_images_from_op_codes(version):
#
#     if not(version == "training" or version == "testing" or version == "benford"):
#         raise Exception(f"Version needs to be train or test, got {version}")
#
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     arr = os.listdir(dir_path + f'/op_codes/{version}/')
#     if '.DS_Store' in arr:
#         arr.remove('.DS_Store')
#
#     max_ = 0
#
#     controlled = []
#     controlled_high = []
#     infected = []
#     infected_high = []
#     for i, file_name in enumerate(arr[:100]):
#
#         file_operations = get_op_code_dictionary()
#
#         with open(dir_path + f'/op_codes/{version}/{file_name}') as json_file:
#             try:
#                 file_data = json.load(json_file)
#             except:
#                 print(file_name)
#
#         line_index = 0
#         for line_index, line in enumerate(file_data['op_codes']):
#             operation = line['operation']
#             if len(operation.split()) > 1:
#                 operation = operation.split()[0]
#
#             if operation in file_operations:
#                 file_operations[operation] += [line_index]
#
#         sample_bin = {x: math.log(1 + (1/x), 10) for x in range(1,10)}
#
#         # reduce_op_code_list_dictionary_to_jump_image(
#         #     file_operations,
#         #     line_index,
#         #     dir_path + f'/image_samples/{file_name.replace(".json", "")}.png'
#         # )
#
#         data_bin = reduce_op_code_list_dictionary_to_leading_digit_bins(
#             file_operations,
#             line_index
#         )
#
#         if not isinstance(data_bin, int):
#             # print(file_name)
#             # print(sample_bin)
#             # print(data_bin)
#             for x in sample_bin:
#                 sample_bin[x] = pow(sample_bin[x] - data_bin[x], 2)
#
#             diff_sum = sum(sample_bin.values())
#             # diff_sum = get_median(sample_bin.values())
#             if diff_sum > max_:
#                 max_ = diff_sum
#
#             if "control" in file_name:
#                 controlled += [diff_sum]
#             else:
#                 infected += [diff_sum]
#             # print(sample_bin)
#
#     print("controlled")
#     mean = sum(controlled) / len(controlled)
#     variance = sum([((x - mean) ** 2) for x in controlled]) / len(controlled)
#     res = variance ** 0.5
#     print(f"mean: {get_mean(controlled)} | median: {get_median(controlled)} | min/max: {get_min(controlled)}/{get_max(controlled)} | standard: {res}")
#     print(controlled_high)
#     print("infected")
#     mean = sum(infected) / len(infected)
#     variance = sum([((x - mean) ** 2) for x in infected]) / len(infected)
#     res = variance ** 0.5
#     print(f"mean: {get_mean(infected)}  | median: {get_median(infected)} | min/max: {get_min(infected)}/{get_max(infected)} | standard: {res}")
#     print(infected_high)
#


def run_benfords_law_test(op_code_directory, sample_size=-1, random_seed=-1):

    if random_seed > 0:
        random.seed(random_seed)

    arr = os.listdir(op_code_directory)
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    controlled = []
    infected = []

    if sample_size < 0:
        sample_size = len(arr)

    random.shuffle(arr)
    arr = arr[:sample_size]

    for i, file_name in enumerate(arr):

        with open(f"{op_code_directory}/{file_name}") as file:
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

        file_operations, line_index = reduce_op_code_list_to_index_list(
            file_data
        )
        data_bin = reduce_op_code_list_dictionary_to_one_jump_leading_digit_bins(
            file_operations,
            line_index
        )

        benford_bin = {x: math.log(1 + (1/x), 10) for x in range(1,10)}

        if isinstance(data_bin, dict):
            diff_sum = difference_by_dictionary(benford_bin, data_bin)

            if "clean" in file_name:
                controlled += [diff_sum]
            else:
                infected += [diff_sum]
            # print(sample_bin)

    print("controlled")
    print_data_statisitics(controlled)
    print("infected")
    print_data_statisitics(infected)

    controlled = np.array(controlled)
    density = stats.gaussian_kde(controlled)
    n, x, _ = plt.hist(controlled, bins=np.linspace(0, 1, 50),
                       histtype=u'step', density=True)
    plt.plot(x, density(x))

    infected = np.array(infected)
    density = stats.gaussian_kde(infected)
    n, y, _ = plt.hist(infected, bins=np.linspace(0, 1, 50),
                       histtype=u'step', density=True)
    plt.plot(y, density(y))
    plt.legend(['Control Distribution', 'Infected Distribution', 'Control Histogram', 'Infected Histogram'])
    plt.show()


def generate_distribution_data(
        op_code_directory,
        func,
        bins=100,
        sample_size=-1,
        random_seed=-1
):
    if random_seed > 0:
        random.seed(random_seed)

    arr = os.listdir(f"{op_code_directory}/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    if sample_size < 0:
        sample_size = len(arr)

    random.shuffle(arr)
    arr = arr[:sample_size]

    for _, file_name in enumerate(arr):
        print(file_name)
        with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

        file_operations, line_index = reduce_op_code_list_to_index_list(
            file_data
        )
        im = np.zeros((len(OP_CODES), bins))
        # im = np.zeros((len(list(set(REVERSED_MAP.values()))), bins))
        # for i, op in enumerate(list(set(REVERSED_MAP.values()))):
        for i, op in enumerate(OP_CODES):
            number_of_operation_instances = len(file_operations[op])
            if number_of_operation_instances > 1:
                for jump_index in range(number_of_operation_instances - 1):
                    jump = file_operations[op][jump_index + 1] - file_operations[op][jump_index]
                    mapped_jump = func(jump / line_index)
                    key = int((mapped_jump * bins) // 1)

                    im[i, key] += 1
                # im[i, :] /= (number_of_operation_instances - 1)
                im[i, :] *= 255.0 / (sum(im[i, :]))
        # im *= 255

        new_image = Image.fromarray(im.astype('float'), 'L')

        file_name_base = file_name.replace('.txt', "")
        new_image.save(f"{op_code_directory}/op_code_distributions/{file_name_base}.png", 'PNG')


def print_data_statisitics(arr):
    mean = sum(arr) / len(arr)
    variance = sum([((x - mean) ** 2) for x in arr]) / len(arr)
    res = variance ** 0.5
    print(
        f"mean: {get_mean(arr)} | median: {get_median(arr)} | min/max: {get_min(arr)}/{get_max(arr)} | standard: {res} | N: {len(arr)}"
    )


def get_sample_distribution(
        op_code_directory,
        func,
        bins=100,
        sample_size=-1
):
    arr = os.listdir(f"{op_code_directory}/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    if sample_size < 0:
        sample_size = len(arr)

    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )

    arr_names = ['clean', 'infected']
    for ii, arr_ in enumerate([clean, infected]):
        arr = arr_[:sample_size]

        linear_multi_sample_distribution_im = np.zeros((len(OP_CODES), bins))
        log_10_multi_sample_distribution_im = np.zeros((len(OP_CODES), bins))
        log_100_multi_sample_distribution_im = np.zeros((len(OP_CODES), bins))

        for _, file_name in enumerate(arr):
            with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
                try:
                    file_data = str(file.read()).split()
                except:
                    print(file_name)

            file_operations, line_index = reduce_op_code_list_to_index_list(
                file_data
            )

            # im = np.zeros((len(list(set(REVERSED_MAP.values()))), bins))
            # for i, op in enumerate(list(set(REVERSED_MAP.values()))):
            for i, op in enumerate(OP_CODES):
                number_of_operation_instances = len(file_operations[op])
                if number_of_operation_instances > 1:
                    for jump_index in range(number_of_operation_instances - 1):
                        jump = file_operations[op][jump_index + 1] - file_operations[op][jump_index]
                        linear_jump = func(jump / line_index)
                        log10_jump = math.log((9 * jump / line_index) + 1, 10)
                        log100_jump = math.log((99 * jump / line_index) + 1, 100)

                        linear_multi_sample_distribution_im[i, int((linear_jump * bins) // 1)] += 1
                        log_10_multi_sample_distribution_im[i, int((log10_jump * bins) // 1)] += 1
                        log_100_multi_sample_distribution_im[i, int((log100_jump * bins) // 1)] += 1

        for i, op in enumerate(OP_CODES):
            linear_multi_sample_distribution_im[i, :] *= 255.0 / (sum(linear_multi_sample_distribution_im[i, :]))
            log_10_multi_sample_distribution_im[i, :] *= 255.0 / (sum(log_10_multi_sample_distribution_im[i, :]))
            log_100_multi_sample_distribution_im[i, :] *= 255.0 / (sum(log_100_multi_sample_distribution_im[i, :]))

        linear_multi_sample_distribution_image = Image.fromarray(linear_multi_sample_distribution_im.astype('float'), 'L')
        linear_multi_sample_distribution_image.save(
            f"{op_code_directory}/op_code_distributions_samples/{arr_names[ii]}_linear_{sample_size}_samples.png", 'PNG')

        log_10_multi_sample_distribution_image = Image.fromarray(log_10_multi_sample_distribution_im.astype('float'), 'L')
        log_10_multi_sample_distribution_image.save(
            f"{op_code_directory}/op_code_distributions_samples/{arr_names[ii]}_log10_{sample_size}_samples.png", 'PNG')

        log_100_multi_sample_distribution_image = Image.fromarray(log_100_multi_sample_distribution_im.astype('float'), 'L')
        log_100_multi_sample_distribution_image.save(
            f"{op_code_directory}/op_code_distributions_samples/{arr_names[ii]}_log100_{sample_size}_samples.png", 'PNG')



if __name__ == "__main__":
    # run_benfords_law_test(
    #     op_code_directory="/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/",
    #     sample_size=10
    # )
    # ops = run_benfords_law_test("benford")
    get_sample_distribution(
        op_code_directory="/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/",
        func=lambda a: a, #math.log((a * 9) + 1, 10),
        bins=100,
        sample_size=1000
    )
    # print(list(set(ops)))

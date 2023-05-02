import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from op_codes import *


def make_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def _find_all_sub_paths(path):
    index = path.rfind("/")
    if index > 0:
        temp = path[:index]
        return [temp] + _find_all_sub_paths(temp)
    return []


def _create_all_possible_sub_directories(base_path, variants):

    directories = list(
        map(
            _find_all_sub_paths,
            [variants[d]['options'][option]['bins'][b]['sub_path']
             for d in variants for option in variants[d]['options'] for b in variants[d]['options'][option]['bins']]
        )
    )
    directories = list(set([item for sublist in directories for item in sublist]))
    directories.sort(key=len)
    for directory in directories:
        make_directory(base_path + "/" + directory)


def get_len(op_code_options, option):
    if option in OP_CODE_CLUSTER:
        return len(list(set(OP_CODE_CLUSTER[option].values())))
    return len(op_code_options[option])


def _get_variant_base_func(base_path, funcs_dictionary, bins, op_code_options):
    def func(sample_size, image_path, iteration):
        return {
            dist: {
                'options': {
                    op_code_option: {
                        'bins': {
                            bin_size: {
                                'image_data': np.zeros((get_len(op_code_options, op_code_option), bin_size)),
                                'sub_path': f"{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                            f"{sample_size}_samples/{image_path}/{iteration}.npy",
                                'path': f"{base_path}/{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                        f"{sample_size}_samples/{image_path}/{iteration}.npy"
                            } for bin_size in bins
                        },
                        'length': get_len(op_code_options, op_code_option)
                    } for op_code_option in op_code_options
                },
                'distribution_func': funcs_dictionary[dist],
            } for dist in funcs_dictionary
        }.copy()
    return func


def _get_split_file_lists(op_code_directory):
    op_code_list = os.listdir(f"{op_code_directory}/op_code_samples/")
    if '.DS_Store' in op_code_list:
        op_code_list.remove('.DS_Store')

    clean_all = list(
        filter(
            lambda x: 'clean' in x,
            op_code_list
        )
    )
    infected_all = list(
        filter(
            lambda x: 'infect' in x,
            op_code_list
        )
    )
    return op_code_list, clean_all, infected_all


def write_jumps_bins_sample(
    op_code_directory,
    funcs_dictionary,
    path,
    op_code_options,
    method,
    bins=None,
    sample_sizes=None,
    iterations=1,
    pruned=False
):

    if method not in ['jump', 'cumulative_share', 'share', 'inverse_jump']:
        raise Exception(f'Unknown Method | {method}')

    path = f"{path}/{method}/{'pruned' if pruned else 'base'}"

    if bins is None:
        bins = [100]

    if sample_sizes is None:
        sample_sizes = [-1]

    variant_func = _get_variant_base_func(
        path,
        funcs_dictionary,
        bins,
        op_code_options
    )

    op_code_list, clean_all, infected_all = _get_split_file_lists(op_code_directory)
    len_op_code_list = len(op_code_list)
    del op_code_list

    dt = datetime.now()
    for iteration in range(iterations):

        for sample_size in sample_sizes:

            if sample_size < 0:
                raise Exception(f'Cannot use negative sample size | sample size: {len_op_code_list}')

            clean = clean_all[(sample_size * iteration):(sample_size * (iteration + 1))]
            infected = infected_all[(sample_size * iteration):(sample_size * (iteration + 1))]

            print(f"Ops: {len_op_code_list}, "
                  f"clean: {len(clean_all)}/{len(clean)}, "
                  f"infected: {len(infected_all)}/{len(infected)}")

            for sets in [
                {
                    'data': clean,
                    'image_path': 'clean'
                },
                {
                    'data': infected,
                    'image_path': 'infected'
                }
            ]:

                arr = sets['data']

                variants = variant_func(
                    sample_size=sample_size,
                    image_path=sets['image_path'],
                    iteration=iteration
                )

                _create_all_possible_sub_directories(
                    base_path=path,
                    variants=variants
                )

                for _, file_name in enumerate(arr):

                    with open(f'{op_code_directory}/op_code_samples/{file_name}') as file:
                        try:
                            file_data = str(file.read()).split()
                        except:
                            print(file_name)

                    file_operations, line_index = reduce_multiple_op_code_lists_to_index_lists(
                        file_data,
                        op_code_options
                    )

                    # TODO - use the index of the op and count how many we have seen
                    #   track bins by line_index which I believe to be the last line number of a file
                    for option, data_ in file_operations.items():
                        op_keys = sorted(list(data_.keys()))
                        for i, op in enumerate(op_keys):
                            number_of_operation_instances = len(file_operations[option][op])
                            file_operations[option][op] = sorted(file_operations[option][op])
                            if method == 'jump' or method == 'inverse_jump':
                                if number_of_operation_instances > 1:
                                    for jump_index in range(number_of_operation_instances - 1):
                                        jump = file_operations[option][op][jump_index + 1] - file_operations[option][op][jump_index]

                                        for d in variants:
                                            for b in variants[d]['options'][option]['bins']:
                                                # TODO make this a function we pass in
                                                mapped_jump = variants[d]['distribution_func'](jump, b)
                                                if mapped_jump < 1:
                                                    key = int((mapped_jump * b) // 1)

                                                    variants[d]['options'][option]['bins'][b]['image_data'][i, key] += 1
                            elif method == 'cumulative_share':
                                for op_line in file_operations[option][op]:
                                    value = op_line / line_index

                                    for d in variants:
                                        for b in variants[d]['options'][option]['bins']:
                                            if value < 1:
                                                key = int(value * b)
                                                variants[d]['options'][option]['bins'][b]['image_data'][i, key:] += 1
                            elif method == 'share':
                                for op_line in file_operations[option][op]:
                                    value = op_line / line_index

                                    for d in variants:
                                        for b in variants[d]['options'][option]['bins']:
                                            if value < 1:
                                                key = int(value * b)
                                                variants[d]['options'][option]['bins'][b]['image_data'][i, key] += 1

                for d in variants:
                    for option in variants[d]['options']:
                        for b in variants[d]['options'][option]['bins']:
                            if method == 'jump':
                                for i in range(variants[d]['options'][option]['length']):
                                    variants[d]['options'][option]['bins'][b]['image_data'][i, :] \
                                        *= 1 / (sum(variants[d]['options'][option]['bins'][b]['image_data'][i, :]) + 1)
                            elif method in ['cumulative_share', 'share', 'inverse_jump']:
                                for i in range(b):
                                    variants[d]['options'][option]['bins'][b]['image_data'][:, i] \
                                        *= 1 / max(sum(variants[d]['options'][option]['bins'][b]['image_data'][:, i]), 1)
                                # for i in range(variants[d]['options'][option]['length']):
                                #     value = sum(variants[d]['options'][option]['bins'][b]['image_data'][i, :])
                                #     variants[d]['options'][option]['bins'][b]['image_data'][i, :] \
                                #         *= 1 / (value if value > 0 else .001)

                            # for i in range(4):
                            #     print(sum(variants[d]['options'][option]['bins'][b]['image_data'][i]))
                            #     print(variants[d]['options'][option]['bins'][b]['image_data'][i])
                            # plt.imshow(variants[d]['options'][option]['bins'][b]['image_data'])
                            # plt.show()

                            with open(variants[d]['options'][option]['bins'][b]['path'], 'wb') as file:
                                np.save(file, variants[d]['options'][option]['bins'][b]['image_data'])

        print(f"{iteration}, {datetime.now().strftime('%I:%M %p')}, {datetime.now() - dt}")
        dt = datetime.now()


if __name__ == "__main__":

    pruned = False

    # ops = ['benign', 'infected', 'union', 'intersection', 'disjoint', 'ratio', 'ratio_a75']
    ops = ['benign', 'infected', 'common_cluster', 'malware_cluster']

    for t in [
        # {
        #     'method': 'inverse_jump',
        #     'distributions': {
        #         'linear': lambda a, b: a / 1000,
        #     }
        # },
        # {
        #     'method': 'cumulative_share',
        #     'distributions': {
        #         'linear': lambda a, b: a / 1000,
        #     }
        # },
        {
            'method': 'jump',
            'distributions': {
                'linear': lambda a, b: a / 1000,
                # 'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
                # 'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
                # 'threshold': lambda a, b: a / b
            }
        },
        # {
        #     'method': 'share',
        #     'distributions': {
        #         'linear': lambda a, b: a / 1000,
        #     }
        # },
    ]:
        temp_ops = {x: OP_CODE_DICT[t['method'] if pruned else 'base'][x] for x in ops}
        write_jumps_bins_sample(
            op_code_directory="/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/",
            sample_sizes=[500],
            path=f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/"
                 f"op_code_distributions_samples/",
            op_code_options=temp_ops,
            bins=[250],
            iterations=10,
            funcs_dictionary=t['distributions'],
            method=t['method'],
            pruned=pruned
        )


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


def _get_variant_base_func(base_path, funcs_dictionary, bins, op_code_options):
    def func(sample_size, image_path, iteration):
        return {
            dist: {
                'options': {
                    op_code_option: {
                        'bins': {
                            bin_size: {
                                'image_data': np.zeros((len(op_code_options[op_code_option]), bin_size)),
                                'sub_path': f"{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                            f"{sample_size}_samples/{image_path}/{iteration}.npy",
                                'path': f"{base_path}/{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                        f"{sample_size}_samples/{image_path}/{iteration}.npy"
                            } for bin_size in bins
                        },
                        'length': len(op_code_options[op_code_option])
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
    bins=None,
    sample_sizes=None,
    iterations=1,
    is_jump=True
):
    if bins is None:
        bins = [100]

    if sample_sizes is None:
        sample_sizes = [-1]

    variant_func = _get_variant_base_func(path, funcs_dictionary, bins, op_code_options)

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
                        for i, op in enumerate(data_):
                            number_of_operation_instances = len(file_operations[option][op])
                            if is_jump:
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
                            else:
                                for op_line in file_operations[option][op]:
                                    value = op_line / line_index

                                    for d in variants:
                                        for b in variants[d]['options'][option]['bins']:
                                            # TODO make this a function we pass in
                                            if value < 1:
                                                key = int(value * b)
                                                variants[d]['options'][option]['bins'][b]['image_data'][i, key:] += 1

                                # for d in variants:
                                #     for b in variants[d]['options'][option]['bins']:
                                #         variants[d]['options'][option]['bins'][b]['image_data'][i] *= \
                                #             (1 / max(variants[d]['options'][option]['bins'][b]['image_data'][i]))
                                #         for k in range(1, b + 1):
                                #             # Divides cell by the number of lines traversed through each bin
                                #             variants[d]['options'][option]['bins'][b]['image_data'][i, k] /= \
                                #                 ((k / b) * line_index)

                for d in variants:
                    for option in variants[d]['options']:
                        for b in variants[d]['options'][option]['bins']:
                            if is_jump:
                                for i in range(variants[d]['options'][option]['length']):
                                    variants[d]['options'][option]['bins'][b]['image_data'][i, :] \
                                        *= 1 / (sum(variants[d]['options'][option]['bins'][b]['image_data'][i, :]) + 1)
                            else:
                                for i in range(b):
                                    variants[d]['options'][option]['bins'][b]['image_data'][:, i] \
                                        *= 1 / max(sum(variants[d]['options'][option]['bins'][b]['image_data'][:, i]), 1)
                                for i in range(variants[d]['options'][option]['length']):
                                    variants[d]['options'][option]['bins'][b]['image_data'][i, :] \
                                        *= 1 / max(max(variants[d]['options'][option]['bins'][b]['image_data'][i, :]), 0.001)

                            # plt.imshow(variants[d]['options'][option]['bins'][b]['image_data'])
                            # plt.show()
                            with open(variants[d]['options'][option]['bins'][b]['path'], 'wb') as file:
                                np.save(file, variants[d]['options'][option]['bins'][b]['image_data'])

        print(f"{iteration}, {datetime.now().strftime('%H:%M')} ,{datetime.now() - dt}")
        dt = datetime.now()


if __name__ == "__main__":


    distributions = {
        'linear': lambda a, b: a / 1000,
        # 'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
        # 'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
        # 'threshold': lambda a, b: a / b
    }

    for t in [
        {
            'folder': 'share',
            'if': False,
            'distributions':{
                'linear': lambda a, b: a / 1000,
            }
        },
        {
            'folder': 'jump',
            'if': True,
            'distributions': {
                'linear': lambda a, b: a / 1000,
                'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
                'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
                'threshold': lambda a, b: a / b
            }
        }
    ]:
        write_jumps_bins_sample(
            op_code_directory="/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/",
            sample_sizes=[500],
            path=f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/"
                 f"op_code_distributions_samples/{t['folder']}",
            op_code_options=OP_CODE_DICT,
            bins=[25, 100],
            iterations=10,
            funcs_dictionary=t['distributions'],
            is_jump=t['if']
        )

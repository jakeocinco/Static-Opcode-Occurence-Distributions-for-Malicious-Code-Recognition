import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from base_functions import *


def make_directory(path):
    # This is just a safe make directory function
    # In this case it does not matter if it already exists
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def _find_all_sub_paths(path):
    # Split a long destionation down into a list of all paths between the end and root.
    # This is just used to make sure all folders are made if needed
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
    # returns a function used to allocate the vectors and variables for each bin version
    def func(sample_size, image_path, iteration):
        return {
            dist: {
                'options': {
                    op_code_option: {
                        'bins': {
                            bin_size: {
                                'image_data': np.zeros((
                                    get_number_of_opcodes_tracked(op_code_options, op_code_option),
                                    bin_size)),
                                'sub_path': f"{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                            f"{sample_size}_samples/{image_path}/{iteration}.npy",
                                'path': f"{base_path}/{op_code_option}/op_codes/{dist}/{bin_size}_bins/"
                                        f"{sample_size}_samples/{image_path}/{iteration}.npy"
                            } for bin_size in bins
                        },
                        'length': get_number_of_opcodes_tracked(op_code_options, op_code_option)
                    } for op_code_option in op_code_options
                },
                'distribution_func': funcs_dictionary[dist],
            } for dist in funcs_dictionary
        }.copy()
    return func


def write_aggregated_distribution_sets(
    op_code_sets,
    bins,
    sample_sizes,
    number_of_distributions=1,
    method='jump',
    pruned=False
):
    # This function creates aggregated distribution sets over a number of different parameters
    #   op_code_sets - this function can take in multiple opcode sets to minimize amount of files that need reread
    #   bins - the number of bins in the distribution, array of ints
    #   sample_sizes - number of samples used to create each bin, array of ints
    #   number_of_distributions - number of distributions to create, int
    #   method - the distribution method, should be jump but others can be tested
    #   pruned - whether or not the pruned version of each opcode should be tested (ignore stick with False)

    # At one point different distributions were tested but this just muddied the test and added extra params
    # Leaving this here since it is still implemented and could be tested further but it is probably not worth it
    funcs_dictionary = {'linear': lambda a, b: a / 1000}

    # Currently the only methods used. Others can be added this is just a safety check
    if method not in ['jump', 'cumulative_share', 'share']:
        raise Exception(f'Unknown Method | {method}')

    distribution_path = f"{DISTRIBUTION_SAMPLE_PATH}/{method}/{'pruned' if pruned else 'base'}"

    # returns a function used to allocate the vectors and variables for each bin version
    # this will be used for distribution
    variant_func = _get_variant_base_func(
        distribution_path,
        funcs_dictionary,
        bins,
        op_code_sets
    )

    # get all files
    op_code_list, clean_all, infected_all = get_split_file_lists()
    len_op_code_list = len(op_code_list)
    del op_code_list

    dt = datetime.now()
    for iteration in range(number_of_distributions):
        for sample_size in sample_sizes:
            if sample_size < 0:
                raise Exception(f'Cannot use negative sample size | sample size: {len_op_code_list}')

            # Different files are selected for different iterations, this ensures that they do not overlap
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

                # getting empty distributions for this iteration
                variants = variant_func(
                    sample_size=sample_size,
                    image_path=sets['image_path'],
                    iteration=iteration
                )

                _create_all_possible_sub_directories(
                    base_path=distribution_path,
                    variants=variants
                )

                for _, file_name in enumerate(arr):

                    with open(file_name) as file:
                        try:
                            file_data = str(file.read()).split()
                        except:
                            print(file_name)

                    file_operations, line_index = reduce_multiple_op_code_lists_to_index_lists(
                        file_data,
                        op_code_sets
                    )

                    #for each opcode set
                    for opcode_option, data_ in file_operations.items():
                        op_keys = sorted(list(data_.keys()))
                        # then for each op code in each set
                        for i, op in enumerate(op_keys):
                            number_of_operation_instances = len(file_operations[opcode_option][op])
                            file_operations[opcode_option][op] = sorted(file_operations[opcode_option][op])

                            # iterate the correct bin (or bins) based on the method being used
                            if method == 'jump':
                                if number_of_operation_instances > 1:
                                    for jump_index in range(number_of_operation_instances - 1):
                                        jump = file_operations[opcode_option][op][jump_index + 1] - file_operations[option][op][jump_index]

                                        for d in variants:
                                            for b in variants[d]['options'][option]['bins']:
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
                            # for each op and bin, divide on the correct axis to create the distribution
                            if method == 'jump':
                                for i in range(variants[d]['options'][option]['length']):
                                    variants[d]['options'][option]['bins'][b]['image_data'][i, :] \
                                        *= 1 / (sum(variants[d]['options'][option]['bins'][b]['image_data'][i, :]) + 1)
                            elif method in ['cumulative_share', 'share']:
                                for i in range(b):
                                    variants[d]['options'][option]['bins'][b]['image_data'][:, i] \
                                        *= 1 / max(sum(variants[d]['options'][option]['bins'][b]['image_data'][:, i]), 1)

                            with open(variants[d]['options'][option]['bins'][b]['path'], 'wb') as file:
                                np.save(file, variants[d]['options'][option]['bins'][b]['image_data'])

        print(f"{iteration}, {datetime.now().strftime('%I:%M %p')}, {datetime.now() - dt}")
        dt = datetime.now()


if __name__ == "__main__":

    # Parameters of script
    pruned = False
    ops = ['benign'] #, 'infected', 'union', 'intersection', 'disjoint', 'ratio', 'ratio_a75']
    method = 'jump'
    num_bins = [25, 100]
    number_of_distributions = 10
    num_samples = [500]

    # The ops change if the data is pruned, so this gets the correct opcode sets
    temp_ops = {x: OP_CODE_DICT[method if pruned else 'base'][x] for x in ops}

    write_aggregated_distribution_sets(
        sample_sizes=num_samples,
        op_code_sets=temp_ops,
        bins=num_bins,
        number_of_distributions=number_of_distributions,
        method=method,
        pruned=pruned
    )



import math
import numpy as np
import sys

from op_codes import *
from base_functions import *

# This script is used to get new opcode sets

def reduce_op_code_list_to_counts(op_code_occurrence_list, file_operations=None):

    temp = {}
    for line_index, line in enumerate(op_code_occurrence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]

        if operation not in temp:
            temp.update({operation: 0})
        temp[operation] += 1

    if file_operations is None:
        file_operations = {}
    else:
        for x in list(set(list(file_operations.keys()) + list(temp.keys()))):
            if x in temp and temp[x] >= 2 and x in file_operations:
                file_operations[x] += 1
            elif x in temp and temp[x] >= 2 and x not in file_operations:
                file_operations.update({x: 1})


def get_top_k_keys(
        k,
        num_files=1000
):
    # This function iterates over num_files * 2 files to get op code occurrence stats
    # This function returns two dictionaries of data. The first is benign the second is malicious
    #   'doc_freq' - the number of executables each opcode occurred in
    #   'counts' - the total number of occurrences of each opcode in all explored files
    #   'keys' - the top k most frequent opcodes by counts in all explored files, in order of count

    _, _clean, _infected = get_split_file_lists()

    total = 2 * num_files

    arr = _clean + _infected

    sets = [
        {
            'data': _clean[:num_files],
            'doc_freq': {},
            'counts': {},
            'keys': []
        },
        {
            'data': _infected[:num_files],
            'doc_freq': {},
            'counts': {},
            'keys': []
        }
    ]
    print()
    for si, s in enumerate(sets):
        for i, file_name in enumerate(s['data']):
            sys.stdout.write("\033[F")
            print(f"[{i + 1 + (si * num_files)}/{total}]")
            with open(file_name) as file:
                try:
                    file_data = str(file.read()).split()
                except:
                    print(file_name)

            for op in list(set(file_data)):
                if op not in s['counts']:
                    s['counts'].update({op: 0})
                s['counts'][op] += file_data.count(op)

            reduce_op_code_list_to_counts(file_data, s['doc_freq'])

        s['doc_freq'] = {k: v for k, v in sorted(s['doc_freq'].items(), key=lambda item: item[1], reverse=True)}

        s['keys'] = list(
            filter(
                lambda z: not ('(' in z or '<' in z or '%' in z or '.' in z or '_' in z or '-' in z),
                list(s['doc_freq'].copy().keys())
            )
        )[:k]

    sets[0].pop('data')
    sets[1].pop('data')

    sys.stdout.write("\033[F")
    return sets[0], sets[1]


def get_union(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    union = list(set(clean['keys'] + infected['keys']))

    if verbose:
        print(f'Union - {k}')
        print(sorted(union))
        print(len(union))

    return sorted(union)


def get_intersection(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    intersection = [x for x in clean['keys'] if x in infected['keys']]

    if verbose:
        print(f'Intersection - {k}')
        print(sorted(intersection))
        print(len(intersection))

    return sorted(intersection)


def get_disjoint(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    disjoint = [x for x in clean['keys'] if x not in infected['keys']] + \
               [x for x in infected['keys'] if x not in clean['keys']]

    if verbose:
        print(f'Disjoint - {k}')
        print(sorted([x for x in clean['keys'] if x not in infected['keys']]))
        print(sorted([x for x in infected['keys'] if x not in clean['keys']]))
        print(sorted(disjoint))
        print(len(disjoint))

    return sorted(disjoint)


def get_ratio(
        k,
        verbose=False,
        clean=None,
        infected=None,
        log_base=10,
        alpha=0.5
):
    def a(x):
        return max(x, .001)

    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    clean = clean['doc_freq']
    infected = infected['doc_freq']
    _set = list(set(list(clean.keys()) + list(infected.keys())))
    def combine(key):
        i = (0 if key not in infected else infected[key])
        c = (0 if key not in clean else clean[key])
        return i + c
    def div_log(key):
        i = (0 if key not in infected else infected[key]) + 0.001
        c = (0 if key not in clean else clean[key]) + 0.001
        return abs(math.log(i / c, log_base))
    combined_counts = np.array([combine(x) for x in _set]).astype(float)
    log_ratio = np.array([div_log(x) for x in _set]).astype(float)

    combined_counts *= 1 / max(combined_counts)
    log_ratio *= 1 / max(log_ratio)

    ratio_dict = {x: (alpha * combined_counts[i]) + ((1 - alpha) * log_ratio[i]) for i, x in enumerate(_set)}
    ratio_dict = {l: v for l, v in sorted(ratio_dict.items(), reverse=True, key=lambda item: item[1])}
    # print(ratio_dict)
    keys = list(ratio_dict.keys())[:k]

    if verbose:
        print(f'Ratio - {k} - alpha: {alpha}')
        print(sorted(keys))
        print(len(keys))

    return keys


if __name__ == "__main__":

    op_code_set_size = 50
    clean, infected = get_top_k_keys(op_code_set_size, 5)

    print(f'Benign - {op_code_set_size}')
    print(clean['keys'])

    print(f'Malicious - {op_code_set_size}')
    print(infected['keys'])

    get_union(
        op_code_set_size,
        clean=clean,
        infected=infected,
        verbose=True
    )
    get_intersection(
        op_code_set_size,
        clean=clean,
        infected=infected,
        verbose=True
    )
    get_disjoint(
        op_code_set_size,
        clean=clean,
        infected=infected,
        verbose=True
    )


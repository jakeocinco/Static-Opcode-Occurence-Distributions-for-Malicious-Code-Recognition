import json


def overwrite_dictionary_entries(n, o):
    if isinstance(n, dict):
        for k in n:
            if k in o:
                o[k] = overwrite_dictionary_entries(n[k], o[k])
            else:
                o.update({k: n[k]})
    else:
        o = n
    return o


def overwrite_json_file(new_file, old_file, temp_file):
    n = json.load(
        open(f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/{new_file}.json", "r")
    )
    o = json.load(
        open(f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/{old_file}.json", "r")
    )

    o = overwrite_dictionary_entries(n, o)

    with open(
            f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/{temp_file}.json",
            "w+"
    ) as f:
        f.write(
            json.dumps(o, indent=4)
        )


if __name__ == "__main__":
    # o = {"base": {0: 'two', 1: 'one'}}
    # y = {"base": {0: 'zero', 2: 'two'}}
    #
    # print(overwrite_dictionary_entries(y, o))
    #
    # o = {"base": {0: {"linear": {"benign": 0, "infected": 1}}}}
    # y = {"base": {0: {"linear": {"benign": .9}}}}
    #
    # print(overwrite_dictionary_entries(y, o))
    # overwrite_json_file(
    #     new_file='jump_9_1682000466',
    #     old_file='jump_9',
    #     temp_file='temp_9'
    # )
    # overwrite_json_file(
    #     new_file='jump_1_1682000524',
    #     old_file='jump_1',
    #     temp_file='temp_1'
    # )
    overwrite_json_file(
        new_file='share_85_1682175354',
        old_file='share_85',
        temp_file='temp_85'
    )


import json

from config import *

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
        open(f"{RESULTS_BASE_PATH}/{new_file}.json", "r")
    )
    o = json.load(
        open(f"{RESULTS_BASE_PATH}/{old_file}.json", "r")
    )

    o = overwrite_dictionary_entries(n, o)

    with open(
            f"{RESULTS_BASE_PATH}/{temp_file}.json","w+"
    ) as f:
        f.write(
            json.dumps(o, indent=4)
        )


if __name__ == "__main__":
    overwrite_json_file(
        new_file='share_85_1682175354',
        old_file='share_85',
        temp_file='temp_85'
    )


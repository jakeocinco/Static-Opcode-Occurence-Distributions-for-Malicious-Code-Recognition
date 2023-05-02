import sys
import subprocess

from test_train_split import get_op_code_array

# '.bfd',
EXTENSIONS = ['.bundle', '.dylib', '.so']
DESTINATION = "/Volumes/T7/MacOS/op_code_samples"

files = {k: [] for k in EXTENSIONS}
checks = {k: 0 for k in EXTENSIONS}

f = open("/Volumes/T7/MacOS/MacOSExecutableList.txt", 'r')
content = f.read().split()


def _map_extensions(x):
    x = x[x.rfind('/'):]
    if '.' in x:
        return False
    else:
        return True
    return x

content = list(
    filter(
        _map_extensions,
        content
    )
)
l = len(content)
print(f"[0/{l}]")

# for file in content:
#     ext = _map_extensions(file)
#     if ext in EXTENSIONS:
#         files[ext] += [file]
#
count = 0
i = 0
for file in content:

    i += 1
    sys.stdout.write("\033[F")
    print(f"[{i + 1}/{l}]")
    try:
        arr = get_op_code_array(
            file, prefix="/Applications/"
        )
        with open(
                f"{DESTINATION}/clean_{count}.txt", "w+"
        ) as f:
            f.write("\n".join(arr))
        count += 1

    except Exception as e:
        print(e)
        pass


        # if not '360.app' in f2:
        #     cmd = f'objdump -d ~/{f2}'
        #     try:
        #         result = subprocess.run([cmd], shell=True, check=True, stdout=subprocess.PIPE)#, stderr=subprocess.DEVNULL)
        #         checks[file] += 1
        #     except:
        #         pass
        #     i += 1
        # if i > 25:
        #     break
    # checks[file] /= 25
#
print(count)
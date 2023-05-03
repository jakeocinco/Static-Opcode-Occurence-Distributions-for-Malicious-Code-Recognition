import json
import sys

from config import  *

SEEDS = [1, 9, 85, 83]
METHOD = 'cumulative_share'
ITERATIONS = range(10)
BINS = [100]
KL = ['dist||x_log']
model = ['mlp_scaled']
pruned = False

if __name__ == "__main__":

    pruned_path = METHOD if pruned else 'base'
    print(f"{METHOD} - Seeds: {SEEDS}, KL: {KL}, Model: {model} {', PRUNED' if pruned_path == 'pruned' else ''}")
    for _op in sys.argv[1:]:

        data = []

        for s in SEEDS:

            results = json.load(
                open(f"{RESULTS_BASE_PATH}/{METHOD}_{s}.json", "r")
            )
            for iteration in results[pruned_path]:
                if int(iteration) in ITERATIONS:
                    for op in results[pruned_path][iteration]['linear']:
                        if op == _op:
                            for bin in results[pruned_path][iteration]['linear'][op]:
                                if int(bin) in BINS:
                                    for kl in results[pruned_path][iteration]['linear'][op][bin]:
                                        if kl in KL:
                                            for m in results[pruned_path][iteration]['linear'][op][bin][kl]:
                                                if m in model:
                                                    # print(iteration, op, bin, kl, m, results[pruned_path][iteration]['linear'][op][bin][kl][m])
                                                    data += [results[pruned_path][iteration]['linear'][op][bin][kl][m]]

        mean = sum(data) / len(data)
        variance = sum([((x - mean) ** 2) for x in data]) / len(data)
        res = variance ** 0.5

        print(f"{_op} - {mean * 100:.1f}, {res * 100:.1f}")



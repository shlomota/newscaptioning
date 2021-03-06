import os
import numpy as np
from glob import glob
import math
import sys


def main(args=sys.argv):
    base = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
    agent_jobs = 2000
    test = ""
    valid = ""

    if 'test' in args:
        test = "test/"

    if 'valid' in args:
        valid = "valid/"

    ids = np.load(f"{base}dbr/{test}{valid}_ids.npy")
    l = glob(f"{base}dbr/{test}{valid}*[!m]")
    l = [os.path.basename(i) for i in l]

    if "test" in l:
        l.remove("test")

    if "valid" in l:
        l.remove("valid")

    if "_ids.npy" in l:
        l.remove("_ids.npy")

    if "_ids_missing.npy" in l:
        l.remove("_ids_missing.npy")

    if "_agents.npy" in l:
        l.remove("_agents.npy")

    if 'update' in args:
        l = np.array(l)
        if test:
            testa = "_test"
        elif valid:
            valida = "_valid"
        else:
            testa = ""
            valida = ""

        np.save(f"{base}_ids{testa}{valida}.npy", l)
        return len(l)

    ids = ids[~np.isin(ids, l)] #remove l from ids
    np.save(f"{base}dbr/{test}{valid}_ids_missing.npy", ids)

    n = len(ids)
    agents_needed = math.ceil(n/agent_jobs)
    print(f"{n} left for {agents_needed} agents with {agent_jobs} each")
    agents = np.arange(agents_needed)
    np.save(f"{base}dbr/{test}{valid}_agents.npy", agents)

    if False:
        test = False
        if test:
            test = "test/"
        else:
            test = ""

        ids = np.load(f"{base}dbr/{test}{valid}_ids.npy")
        l = glob(f"{base}dbr/{test}{valid}*m")
        l = [os.path.basename(i)[:-1] for i in l]
        ids = ids[~np.isin(ids, l)]  # remove l from ids
        if test:
            test = "_test"
        elif valid:
            test = "_valid"

        np.save(f"{base}/_missing_mask{test}.npy", ids)


if __name__ == '__main__':
    main()

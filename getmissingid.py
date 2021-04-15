import os
import numpy as np
from glob import glob
import math
import sys

base = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"

agent_jobs = 5000

test = ""
mask = "[!m]"

if 'test' in sys.argv:
    test = "test/"

if 'mask' in sys.argv:
    mask = "m"

ids = np.load(f"{base}dbr/{test}_ids.npy")
l = glob(f"{base}dbr/{test}*{mask}")
l = [os.path.basename(i) for i in l]
if "test" in l:
    l.remove("test")
if "_ids.npy" in l:
    l.remove("_ids.npy")
if "_ids_missing.npy" in l:
    l.remove("_ids_missing.npy")
if "_agents.npy" in l:
    l.remove("_agents.npy")
if "_ids_missing_mask.npy" in l:
    l.remove("_ids_missing_mask.npy")

ids = ids[~np.isin(ids, l)] #remove l from ids
np.save(f"{base}dbr/{test}_ids_missing.npy", ids)

if not ('mask' in sys.argv):
    n = len(ids)
    agents_needed = math.ceil(n/agent_jobs)
    print(f"{n} left for {agents_needed} agents with {agent_jobs} each")
    agents = np.arange(agents_needed)
    np.save(f"{base}dbr/{test}_agents.npy", agents)


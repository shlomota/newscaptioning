import os
import numpy as np
from glob import glob
import math
import sys

agent_jobs = 5000

test = ""

if 'test' in sys.argv:
    test = "test/"

ids = np.load(f"dbr/{test}_ids.npy")
l = glob(f"dbr/{test}*[!m]")
l = [os.path.basename(i) for i in l]
if "test" in l:
    l.remove("test")
if "_ids.npy" in l:
    l.remove("_ids.npy")
if "_ids_missing.npy" in l:
    l.remove("_ids_missing.npy")
if "_agents.npy" in l:
    l.remove("_agents.npy")

ids = ids[~np.isin(ids, l)] #remove l from ids
np.save(f"dbr/{test}_ids_missing.npy", ids)

n = len(ids)
agents_needed = math.ceil(n/agent_jobs)
print(f"{n} left for {agents_needed} agents with {agent_jobs} each")
agents = np.arange(agents_needed)
np.save(f"dbr/{test}_agents.npy", agents)


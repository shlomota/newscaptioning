import os
import numpy as np
from glob import glob
import math

agent_jobs = 5000

ids = np.load("dbr/_ids.npy")
l = glob("dbr/*[!m]")
l = [os.path.basename(i) for i in l]
l.remove("test")
l.remove("_ids.npy")
l.remove("_ids_missing.npy")
l.remove("_agents.npy")

ids = ids[~np.isin(ids, l)] #remove l from ids
np.save("dbr/_ids_missing.npy", ids)

n = len(ids)
agents_needed = math.ceil(n/agent_jobs)
print(f"{n} left for {agents_needed} agents with {agent_jobs} each")
agents = np.arange(agents_needed)
np.save("dbr/_agents.npy", agents)


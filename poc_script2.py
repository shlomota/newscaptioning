import os
import time
from tell.commands.evaluate import evaluate_from_file as eff

TAT_FOLDER = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
OUTS_FOLDER = "outs"
model = "11_new2"

YAML = "expt/nytimes/{}/config.yaml".format(model)
YAML = os.path.join(TAT_FOLDER,YAML)

SER = "expt/nytimes/{}/serialization/best.th".format(model)
SER = os.path.join(TAT_FOLDER,SER)

G_JSONL = "expt/nytimes/{}/serialization/generations.jsonl".format(model)
G_JSONL = os.path.join(TAT_FOLDER, G_JSONL)

if os.path.exists(G_JSONL):
    os.remove(G_JSONL)

eff(YAML, SER, device=0)

os.rename(G_JSONL, os.path.join(TAT_FOLDER, OUTS_FOLDER, f"generations_{round(time.time())}.jsonl"))
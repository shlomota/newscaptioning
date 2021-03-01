import os
from tell.commands.evaluate import evaluate_from_file as eff

TAT_FOLDER = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/transform-and-tell/"
model = "10_new"

YAML = "expt/nytimes/{}/config.yaml".format(model)
YAML = os.path.join(TAT_FOLDER,YAML)

SER = "expt/nytimes/{}/serialization/best.th".format(model)
SER = os.path.join(TAT_FOLDER,SER)

G_JSONL = "expt/nytimes/{}/serialization/generations.jsonl".format(model)
G_JSONL = os.path.join(TAT_FOLDER, G_JSONL)

if os.path.exists(G_JSONL):
    os.remove(G_JSONL)

eff(YAML, SER, device=0)

import os
import datetime
from tell.commands.evaluate import evaluate_from_file as eff
import shutil

TAT_FOLDER = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
OUTS_FOLDER = "outs"
# model = "9_transformer_objects"
model = "BMTestModel"

YAML = "expt/nytimes/{}/config.yaml".format(model)
YAML = os.path.join(TAT_FOLDER,YAML)

SER = "expt/nytimes/{}/serialization/best.th".format(model)
SER = os.path.join(TAT_FOLDER,SER)

G_JSONL = "expt/nytimes/{}/serialization/generations.jsonl".format(model)
G_JSONL = os.path.join(TAT_FOLDER, G_JSONL)

if os.path.exists(G_JSONL):
    os.remove(G_JSONL)

eff(YAML, SER, device=0)

now = datetime.datetime.now()
shutil.copyfile(G_JSONL, os.path.join(TAT_FOLDER, OUTS_FOLDER, f"generations_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.jsonl"))

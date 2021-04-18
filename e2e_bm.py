import os
import datetime
from tell.commands.evaluate import evaluate_from_file as eff
import shutil

TAT_FOLDER = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
OUTS_FOLDER = "outs"
model = "13_bm"

YAML = f"expt/nytimes/{model}/config.yaml"
YAML = os.path.join(TAT_FOLDER, YAML)
# expt/nytimes/BM/serialization_sum_good/best.th
SER = f"expt/nytimes/{model}/serialization/best.th"
# SER = "expt/nytimes/{}/serialization_sum_good/best.th".format(model)
SER = os.path.join(TAT_FOLDER, SER)

G_JSONL = f"expt/nytimes/{model}/serialization/generations.jsonl"
G_JSONL = os.path.join(TAT_FOLDER, G_JSONL)


# if os.path.exists(G_JSONL):
#     os.remove(G_JSONL)


def run():
    eff(YAML, SER, device=0)


need_to_run = True
i = 0
while need_to_run:
    print(f"While...{i}")
    try:
        eff(YAML, SER, device=0)
    except RuntimeError as e:
        print(e)
        continue

    need_to_run = False

now = datetime.datetime.now()
shutil.copyfile(G_JSONL, os.path.join(TAT_FOLDER, OUTS_FOLDER,
                                      f"generations_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.jsonl"))

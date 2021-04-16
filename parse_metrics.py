import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd

BASE_PATH = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BM/serialization_40_256/")


# SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BMRel/serialization_mean/")

def get_epoch(filename):
    return int(filename.split("_")[-1].split(".")[0])


files = glob.glob(os.path.join(SERIALIZATION_DIR, "metrics_epoch*"))
files_with_epochs = [(filename, get_epoch(filename)) for filename in files]
files_with_epochs = sorted(files_with_epochs, key=lambda x: x[1])
sorted_files = [file[0] for file in files_with_epochs]

training_loss = []
validation_loss = []
for file in sorted_files:
    with open(file, "r") as f:
        data = json.load(f)
    training_loss.append(data["training_loss"])
    validation_loss.append(data["validation_loss"])

plt.plot(range(len(sorted_files)), training_loss, label="train")
plt.plot(range(len(sorted_files)), validation_loss, label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()

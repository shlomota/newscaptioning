import sys
import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd

BASE_PATH = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/expt/nytimes"
DIR = "BM/serialization_40_256/"
DIR = "BMRel/serialization_mean_100_2048/"

if len(sys.argv) > 1:
    DIR = sys.argv[1]

SERIALIZATION_DIR = os.path.join(BASE_PATH, DIR)
if not os.path.exists(SERIALIZATION_DIR):
    raise Exception("yo wut?")

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

if not 'noplot' in sys.argv:
    plt.plot(range(len(sorted_files)), training_loss, label="train")
    plt.plot(range(len(sorted_files)), validation_loss, label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()
else:
    print(len(sorted_files))
    print(training_loss)
    print(validation_loss)

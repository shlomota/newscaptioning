from tell.commands.train import train_model_from_file
import shutil
import os
import sys

idd = ""
if len(sys.argv) > 1:
    idd = sys.argv[1]

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
parameter_filename = "expt/nytimes/BM2/config.yaml"
serialization_dir = base_path + f"expt/nytimes/BM2/serialization{idd}"

LOSSLOG = os.path.join(base_path, f'BMM{idd}.log')

if os.path.exists(serialization_dir):
    shutil.rmtree(serialization_dir)

if os.path.exists(LOSSLOG):
    os.remove(LOSSLOG)

train_model_from_file(parameter_filename, serialization_dir)
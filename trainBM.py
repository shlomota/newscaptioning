from tell.commands.train import train_model_from_file
import os
import shutil

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
parameter_filename = "expt/nytimes/BM/config.yaml"
serialization_dir = base_path + "expt/nytimes/BM/serialization_40"

if os.path.exists(serialization_dir):
    shutil.rmtree(serialization_dir)
train_model_from_file(parameter_filename, serialization_dir)
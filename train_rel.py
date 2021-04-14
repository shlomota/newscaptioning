from tell.commands.train import train_model_from_file
import shutil
import os

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
parameter_filename = "expt/nytimes/BMRel/config.yaml"
serialization_dir = base_path + "expt/nytimes/BMRel/serialization_mean"

LOSSLOG = os.path.join(base_path, 'BM_rel2.log')

if os.path.exists(serialization_dir):
    shutil.rmtree(serialization_dir)

if os.path.exists(LOSSLOG):
    os.remove(LOSSLOG)

train_model_from_file(parameter_filename, serialization_dir)

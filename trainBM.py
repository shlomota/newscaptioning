from tell.commands.train import train_model_from_file
import shutil
import os

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
parameter_filename = "expt/nytimes/BM/config.yaml"
serialization_dir = base_path + "expt/nytimes/BM/serialization_sum"

print("directory content:", os.listdir(serialization_dir))
shutil.rmtree(serialization_dir)
print("after remove")
train_model_from_file(parameter_filename, serialization_dir)

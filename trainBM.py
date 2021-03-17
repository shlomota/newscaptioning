from tell.commands.train import train_model_from_file
import shutil

parameter_filename = "expt/nytimes/BM/config.yaml"
serialization_dir = "expt/nytimes/BM/serialization"

shutil.rmtree(serialization_dir)
train_model_from_file(parameter_filename, serialization_dir)
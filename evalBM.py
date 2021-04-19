from tell.commands.bm_evaluate import get_model_from_file, evaluate
import shutil
import os

from tell.commands.train import train_model_from_file

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
parameter_filename = "expt/nytimes/BM/config.yaml"
serialization_dir = base_path + "expt/nytimes/BM/serialization"

print("directory content:", os.listdir(serialization_dir))
shutil.rmtree(serialization_dir)
print("after remove")
train_model_from_file(parameter_filename, serialization_dir)
# model = get_model_from_file(parameter_filename, serialization_dir)
# evaluate(model, [])

# def evaluate(model: Model,
#              instances: Iterable[Instance],
#              data_iterator: DataIterator,
#              cuda_device: int,
#              serialization_dir: str,
#              eval_suffix: str,
#              batch_weight_key: str) -> Dict[str, Any]:


from allennlp.commands.train import train_model
import yaml
from allennlp.common.params import Params
from allennlp.models import Model
import shutil

params_file = "config.yaml"
serialization_dir = "serialization"


def train_model_from_file(params_file: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          cache_directory: str = None,
                          cache_prefix: str = None) -> Model:
    with open(params_file) as f:
        file_dict = yaml.safe_load(f)

    params = Params(file_dict)

    return train_model(params,
        serialization_dir,
        file_friendly_logging,
        recover,
        force,
        cache_directory, cache_prefix)

def main():
    shutil.rmtree(serialization_dir)
    train_model_from_file(params_file, serialization_dir)

if __name__ == '__main__':
    main()
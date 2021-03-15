from allennlp.commands.train import train_model
from data.dataset_readers.reader import BMReader
import yaml
from allennlp.common.params import Params
from allennlp.models import Model
import shutil
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from tell.commands.train import yaml_to_params


params_file = "config.yaml"
serialization_dir = "serialization"

config_path = 'config.yaml'
config = yaml_to_params(config_path, overrides='')

tokenizer = Tokenizer.from_params(config.get('dataset_reader').get('tokenizer'))
indexer_params = config.get('dataset_reader').get('token_indexers')
token_indexers = {k: TokenIndexer.from_params(p)
                       for k, p in indexer_params.items()}


reader = BMReader(tokenizer, token_indexers, image_dir="../data/nytimes/images_processed")
# reader._read()

# if __name__ == '__main__':
#     main()
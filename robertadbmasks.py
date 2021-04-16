import pymongo
from tqdm import tqdm
import torch
import numpy as np
from glob import glob
import sys
import os

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.fields import TextField
from tell.data.fields import ListTextField
from tell.commands.train import yaml_to_params

base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/dbr/"

config_path = "expt/nytimes/BM2/config.yaml"
config = yaml_to_params(config_path, overrides='')

client = pymongo.MongoClient(host='nova', port=27017)
db = client.nytimes

split = 'train'
reverse = False
full_search = False
overwrite = False

if 'train' in sys.argv:
    split = 'train'

if 'test' in sys.argv:
    split = 'test'

if 'r' in sys.argv:
    reverse = True

if 'fs' in sys.argv:
    full_search = True

if 'ow' in sys.argv:
    overwrite = True

if split not in ['train', 'test']:
    raise Exception('w0t?')

if split == 'test':
    base_path += 'test/'

articles = db.articles.find({'split': split}, projection=['_id']).sort('_id', pymongo.ASCENDING)

tokenizer = Tokenizer.from_params(config.get('dataset_reader').get('tokenizer'))
indexer_params = config.get('dataset_reader').get('token_indexers')
token_indexer = {k: TokenIndexer.from_params(p) for k, p in indexer_params.items()}
vocab = Vocabulary.from_params(config.get('vocabulary'))

roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.large')
roberta.eval()

ids = np.load(base_path+'_ids.npy')
#ids = np.load("/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/_missing_mask.npy")

if reverse:
    ids = ids[::-1]

#['_id', 'web_url', 'snippet', 'lead_paragraph', 'abstract', 'print_page', 'blog', 'source', 'multimedia', 'headline', \
# 'keywords', 'pub_date', 'document_type', 'news_desk', 'section_name', 'subsection_name', 'byline',\
# 'type_of_material', 'word_count', 'slideshow_credits', 'scraped', 'parsed', 'error', 'image_positions',\
# 'parsed_section', 'n_images', 'language', 'parts_of_speech', 'n_images_with_faces', 'detected_face_positions', 'split']

projection = ['_id', 'parsed_section']

if not overwrite:
    sfrom = True
    l = [os.path.basename(id)[:-1] for id in glob(base_path+"*m")]
print(len(ids))

for aid in tqdm(ids):
    if not overwrite:
        if full_search:
            l = [os.path.basename(id) for id in glob(base_path + "*m")]
            if aid in l:
                continue
        else:
            if sfrom:
                if aid in l:
                    continue
                else:
                    sfrom = False
                    del l

    a = db.articles.find_one({'_id': {'$eq': aid}}, projection=projection)
    sections = a['parsed_section']
    paragraphs = [p for p in sections if p['type'] == 'paragraph']

    if not len(paragraphs):
        continue

    tokens = [tokenizer.tokenize(c['text']) for c in paragraphs]
    context = ListTextField([TextField(p, token_indexer) for p in tokens])
    context.index(vocab)
    context = context.as_tensor(context.get_padding_lengths())

    torch.save(context['roberta_copy_masks'], base_path + aid+"m")

articles.close()

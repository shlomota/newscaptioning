import logging
import os
import random
import re
from typing import Dict

import numpy as np
import pymongo
import shutil
import torch
from allennlp.data import DataIterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import pandas as pd

from tell.commands.bm_evaluate import get_model_from_file, evaluate
from tell.commands.train import yaml_to_params
from tell.data.fields import ImageField, ListTextField

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SPACE_NORMALIZER = re.compile(r"\s+")
split = 'test'


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


CONFIG_PATH = "expt/nytimes/BM2/config.yaml"
BASE_PATH = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
# SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BM/serialization_sum_good/")
SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BM2/serializationNarch2_1024_512_100/best.th")

def get_bmmodel():
    # print("directory content:", os.listdir(SERIALIZATION_DIR))
    # shutil.rmtree(SERIALIZATION_DIR)
    # print("after remove")
    overrides = """{"vocabulary":
                     {"type": "roberta",
                      "directory_path": "./expt/vocabulary"}"""
    return get_model_from_file(CONFIG_PATH, SERIALIZATION_DIR)

model = get_bmmodel()
image_dir = '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/data/nytimes/images_processed'
preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
client = MongoClient(host='nova', port=27017)
db = client.nytimes

base = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
splitn = ''
if split == 'test':
    splitn = '_test'
elif split == 'valid':
    splitn = '_valid'

ids = np.array([])
while not len(ids):  # is someone else reading/writing ? Wait a bit...
    try:
        ids = np.load(f"{base}_ids{splitn}.npy")

    except Exception:
        sleep(1)

np.random.shuffle(ids)

projection = ['_id', 'parsed_section.type', 'parsed_section.text',
              'parsed_section.hash', 'parsed_section.parts_of_speech',
              'image_positions']

for article_id in ids[:10]:
    article = db.articles.find_one({'_id': {'$eq': article_id}}, projection=projection)
    sections = article['parsed_section']
    image_positions = article['image_positions']
    for pos in image_positions:
        paragraphs = [p for p in sections if p['type'] == 'paragraph']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image_path = os.path.join(image_dir, f"{sections[pos]['hash']}.jpg")
        image = Image.open(image_path)
        iff = ImageField(image, preprocess)
        iff = iff.as_tensor(iff.get_padding_lengths()).unsqueeze(0)
        iff = iff.to(device)

        results = model.forward(aid=[article_id], split=[split], label=torch.ones((1, len(paragraphs))).to(device),
                                     image=iff, caption=torch.tensor([1]))
        probs = torch.exp(results["probs"])
        print(probs)


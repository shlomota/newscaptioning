import logging
import os
import random
import re
from typing import Dict, List
from time import sleep
import math

import numpy as np
import pymongo
import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField, ListField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from tell.data.fields import ImageField, ListTextField

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@DatasetReader.register('BMReader')
class BMReader(DatasetReader):
    """Read from the New York Times dataset.

    See the repo README for more instruction on how to download the dataset.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 use_caption_names: bool = True,
                 use_objects: bool = False,
                 n_faces: int = None,
                 lazy: bool = True,
                 articles_num: int = -1,
                 use_first: bool = True,
                 sort_BM: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.nytimes
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices
        self.articles_num = articles_num

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')

        # load npy ids
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

        self.rs.shuffle(ids)
        print(f"found {len(ids)} article ids")

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']

        if self.articles_num == -1:
            self.articles_num = len(ids)

        print(f'articles num: {self.articles_num}')

        for article_id in ids[:self.articles_num]:
            article = self.db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']

            paragraphs = []
            named_entities = set()


            paragraphs += [p for p in sections if p['type'] == 'paragraph']
            paragraphs_texts = [p["text"] for p in paragraphs]

            tokenized_corpus = [doc.split(" ") for doc in paragraphs_texts]
            bm25 = BM25Okapi(tokenized_corpus)


            image_positions = article['image_positions']
            for pos in image_positions:
                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                #label generation
                query = caption
                tokenized_query = query.split(" ")
                paragraphs_scores = bm25.get_scores(tokenized_query)


                image_id = f'{article_id}_{pos}'

                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                face_embeds = np.array([[]])

                obj_feats = None

                for i, _ in enumerate(paragraphs):
                    yield self.article_to_instance(
                        article_id, i, paragraphs_scores[i], named_entities, image, caption, image_path,
                        article['web_url'], pos, face_embeds, obj_feats, image_id)
                # yield self.article_to_instance(
                #         paragraphs,paragraphs_scores, named_entities, image, caption, image_path,
                #         article['web_url'], pos, face_embeds, obj_feats, image_id)

    # def article_to_instance(self, paragraphs, paragraphs_scores, named_entities, image, caption,
    #                         image_path, web_url, pos, face_embeds, obj_feats, image_id) -> Instance:

    def article_to_instance(self, article_id, i, paragraph_score, named_entities, image, caption,
                            image_path, web_url, pos, face_embeds, obj_feats, image_id, split) -> Instance:

        caption_tokens = self._tokenizer.tokenize(caption)
        name_token_list = [self._tokenizer.tokenize(n) for n in named_entities]

        fields = {
            'aid': MetadataField(article_id),
            'split': MetadataField(split),
            'index': MetadataField(i),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
            'label': ArrayField(np.array([paragraph_score]))
        }

        return Instance(fields)

    def _get_named_entities(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids

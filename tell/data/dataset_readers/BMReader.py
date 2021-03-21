import logging
import os
import random
import re
from typing import Dict, List

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

        #TODO: restore this
        # sample_cursor = self.db.articles.find({
        #     'split': split,
        # }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        # ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        # sample_cursor.close()
        # self.rs.shuffle(ids)

        # TODO: just for debug
        article = self.db.articles.find_one({
            'split': split,
        }, projection=['_id'])
        ids = np.array([article['_id']])





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

            for p in paragraphs:
                named_entities |= self._get_named_entities(p)
            named_entities = sorted(named_entities)

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

                # apply softmax
                #TODO: restore?
                # paragraphs_scores = np.exp(paragraphs_scores)
                # paragraphs_scores = paragraphs_scores / paragraphs_scores.sum(0)


                image_id = f'{article_id}_{pos}'

                if self.n_faces is not None:
                    n_persons = self.n_faces
                elif self.use_caption_names:
                    n_persons = len(self._get_person_names(sections[pos]))
                else:
                    n_persons = 4

                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                if 'facenet_details' not in sections[pos] or n_persons == 0:
                    face_embeds = np.array([[]])
                else:
                    face_embeds = sections[pos]['facenet_details']['embeddings']
                    # Keep only the top faces (sorted by size)
                    face_embeds = np.array(face_embeds[:n_persons])

                obj_feats = None
                if self.use_objects:
                    obj = self.db.objects.find_one(
                        {'_id': sections[pos]['hash']})
                    if obj is not None:
                        obj_feats = obj['object_features']
                        if len(obj_feats) == 0:
                            obj_feats = np.array([[]])
                        else:
                            obj_feats = np.array(obj_feats)
                    else:
                        obj_feats = np.array([[]])

                for i, _ in enumerate(paragraphs):
                    yield self.article_to_instance(
                        paragraphs_texts[i],paragraphs_scores[i], named_entities, image, caption, image_path,
                        article['web_url'], pos, face_embeds, obj_feats, image_id)
                # yield self.article_to_instance(
                #         paragraphs,paragraphs_scores, named_entities, image, caption, image_path,
                #         article['web_url'], pos, face_embeds, obj_feats, image_id)

    # def article_to_instance(self, paragraphs, paragraphs_scores, named_entities, image, caption,
    #                         image_path, web_url, pos, face_embeds, obj_feats, image_id) -> Instance:

    def article_to_instance(self, paragraph, paragraph_score, named_entities, image, caption,
                            image_path, web_url, pos, face_embeds, obj_feats, image_id) -> Instance:
        # context = ' BLABLA '.join([p["text"] for p in paragraphs]).strip()
        context = paragraph

        # context_tokens = [self._tokenizer.tokenize(p["text"]) for p in paragraphs]
        # context_tokens = [self._tokenizer.tokenize(p["text"]) for p in paragraphs]
        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)
        name_token_list = [self._tokenizer.tokenize(n) for n in named_entities]

        if name_token_list:
            name_field = [TextField(tokens, self._token_indexers)
                          for tokens in name_token_list]
        else:
            stub_field = ListTextField(
                [TextField(caption_tokens, self._token_indexers)])
            name_field = stub_field.empty_field()

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            # 'context': ListTextField([TextField(p, self._token_indexers) for p in context_tokens]),
            # 'context': ListTextField(context_tokens),
            'names': ListTextField(name_field),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
            'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
            'label': ArrayField(np.array([paragraph_score]))
            # 'labels': ArrayField(paragraphs_score)
        }

        if obj_feats is not None:
            fields['obj_embeds'] = ArrayField(obj_feats, padding_value=np.nan)

        metadata = {'context': context,
                    'caption': caption,
                    'names': named_entities,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos,
                    'image_id': image_id,
                    'label': paragraph_score}
        fields['metadata'] = MetadataField(metadata)

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

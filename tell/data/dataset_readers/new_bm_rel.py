import logging
import os
import random
import re
from typing import Dict
from time import sleep

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


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


CONFIG_PATH = "expt/nytimes/BMRel/config.yaml"
BASE_PATH = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/"
# SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BM/serialization_sum_good/")
SERIALIZATION_DIR = os.path.join(BASE_PATH, "expt/nytimes/BMRel/serialization_mean_100_2048/best.th")


# FULL PATH -  /a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/expt/nytimes/BM/serialization_1/


def get_bmmodel():
    # print("directory content:", os.listdir(SERIALIZATION_DIR))
    # shutil.rmtree(SERIALIZATION_DIR)
    # print("after remove")
    overrides = """{"vocabulary":
                     {"type": "roberta",
                      "directory_path": "./expt/vocabulary"}"""
    return get_model_from_file(CONFIG_PATH, SERIALIZATION_DIR)


@DatasetReader.register('new_bm_rel')
class NewBMRelReader(DatasetReader):
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
        print(mongo_host)
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
        self.use_first = use_first
        self.sort_BM = sort_BM
        self.model = get_bmmodel()

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

        print('split', split)

        ids = np.array([])
        while not len(ids):  # is someone else reading/writing ? Wait a bit...
            try:
                ids = np.load(f"{base}_ids{splitn}.npy")

            except Exception:
                sleep(1)

        np.random.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'image_positions']
        if self.articles_num == -1:
            self.articles_num = len(ids)

        print(f'articles num: {self.articles_num}')

        for article_id in ids[:self.articles_num]:
            article = self.db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']
            image_positions = article['image_positions']
            for pos in image_positions:
                image_id = f'{article_id}_{pos}'
                paragraphs = []
                named_entities = set()
                n_words = 0
                title = ''
                if 'main' in article['headline']:
                    title = article['headline']['main'].strip()

                if title:
                    paragraphs.append(title)
                    named_entities.union(
                        self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                if self.n_faces is not None:
                    n_persons = self.n_faces
                elif self.use_caption_names:
                    n_persons = len(self._get_person_names(sections[pos]))
                else:
                    n_persons = 4

                # old way - 1st + around image
                pi_chosen = []
                before = []
                after = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        named_entities |= self._get_named_entities(section)
                        pi_chosen.append(k)
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        named_entities |= self._get_named_entities(sections[i])
                        n_words += len(self.to_token_ids(text))
                        pi_chosen.append(i)
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
                        named_entities |= self._get_named_entities(sections[j])
                        n_words += len(self.to_token_ids(text))
                        pi_chosen.append(j)
                    j += 1

                    if n_words >= 510 or (i <= k and j >= len(sections)):
                        break

                paragraphs = paragraphs + before + after
                named_entities = sorted(named_entities)
                pi_chosen.sort()

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

                '''yield self.article_to_instance(
                    paragraphs, named_entities, image, caption, image_path,
                    article['web_url'], pos, face_embeds, obj_feats, image_id, pi_chosen, gen_type=1)
                '''

                # 'new' way of sending every paragraph
                paragraphs = [p for p in sections if p['type'] == 'paragraph']

                # todo: restore
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                results = self.model.forward(aid=[article_id] * len(paragraphs), index1=range(len(paragraphs)),
                                             index2=range(len(paragraphs)), split=[split],
                                             label=torch.tensor([1]).to(device),
                                             image=ImageField(image, self.preprocess))
                scores = results["score0"]
                paragraphs_scores = torch.stack(scores).to(device="cpu").numpy()

                # paragraphs_texts = [p["text"] for p in paragraphs]
                # tokenized_corpus = [doc.split(" ") for doc in paragraphs_texts]
                # bm25 = BM25Okapi(tokenized_corpus)
                # query = caption
                # tokenized_query = query.split(" ")
                # paragraphs_scores = bm25.get_scores(tokenized_query)

                df = pd.DataFrame(columns=["paragraph", "score", "i"])
                df.paragraph = paragraphs
                df.score = paragraphs_scores
                df.i = list(range(len(paragraphs)))
                # df.score = list(range(len(paragraphs)))
                df = df.sort_index(ascending=True)

                if self.sort_BM:
                    sorted_df = df.sort_values("score", ascending=False)
                    thresh = 0
                    text_count = 0
                    if title:
                        text_count = len(title)

                    for index, row in (sorted_df.iterrows()):
                        text_count += len(row.paragraph["text"])
                        if text_count > 512:
                            thresh = row.score
                            break
                    if self.use_first:
                        df = df[(df.score >= thresh) | (df.i == 0)]
                    else:
                        df = df[df.score >= thresh]
                else:
                    df = df.sort_values("score", ascending=False)

                # sort df
                n_words = 0
                named_entities = set()
                sorted_paragraphs = []
                pi_chosen = []

                if title:
                    sorted_paragraphs.append(title)
                    named_entities = named_entities.union(
                        self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

                if self.use_first:
                    fp = df[df.i == 0]
                    fp = fp["paragraph"].values.tolist()
                    if fp:
                        fp = fp[0]
                        text = fp["text"]
                        n_words += len(self.to_token_ids(text))
                        sorted_paragraphs.append(text)
                        named_entities |= self._get_named_entities(fp)
                        pi_chosen.append(0)
                    else:
                        print(f"?!{len(paragraphs)}")

                if n_words < 510:
                    if self.use_first:  # if used 1st paragraph, don't reuse
                        df = df[df['i'] != 0]

                    for p, i in zip(df["paragraph"].values.tolist(), df["i"].values.tolist()):
                        text = p["text"]
                        n_words += len(self.to_token_ids(text))
                        sorted_paragraphs.append(text)
                        named_entities |= self._get_named_entities(p)
                        pi_chosen.append(i)
                        if n_words >= 510:
                            break

                named_entities = sorted(named_entities)
                yield self.article_to_instance(
                    sorted_paragraphs, named_entities, image, caption, image_path,
                    article['web_url'], pos, face_embeds, obj_feats, image_id, pi_chosen, gen_type=2)

    def article_to_instance(self, paragraphs, named_entities, image, caption,
                            image_path, web_url, pos, face_embeds, obj_feats, image_id, pi_chosen,
                            gen_type) -> Instance:
        context = '\n'.join(paragraphs).strip()

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
            'names': ListTextField(name_field),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
            'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
        }

        if obj_feats is not None:
            fields['obj_embeds'] = ArrayField(obj_feats, padding_value=np.nan)

        metadata = {'context': context,
                    'caption': caption,
                    'names': named_entities,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos,
                    'pi_chosen': pi_chosen,
                    'gen_type': gen_type,
                    'image_id': image_id}
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

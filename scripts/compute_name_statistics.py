"""Get articles from the New York Times API.

Usage:
    compute_name_statistics.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].

"""
import pickle
from collections import Counter
from datetime import datetime

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'host': str
    })
    args = schema.validate(args)
    return args


def compute_nytimes(client):
    nytimes = client.nytimes
    start = datetime(2000, 1, 1)
    end = datetime(2019, 5, 1)

    caption_counter = Counter()
    context_counter = Counter()

    article_cursor = nytimes.articles.find({
        'split': 'train',
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        get_proper_names(article['headline'], context_counter)

        sections = article['parsed_section']
        for section in sections:
            if section['type'] == 'caption':
                get_proper_names(section, caption_counter)
            elif section['type'] == 'paragraph':
                get_proper_names(section, context_counter)
            else:
                raise ValueError(f"Unknown type: {section['type']}")

    counters = {
        'caption': caption_counter,
        'context': context_counter,
    }
    with open('./data/nytimes/name_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)


def compute_goodnews(client):
    goodnews = client.goodnews

    caption_counter = Counter()
    context_counter = Counter()

    sample_cursor = goodnews.splits.find(
        {'split': 'train'}, no_cursor_timeout=True).batch_size(128)

    done_article_ids = set()
    for sample in tqdm(sample_cursor):
        if sample['article_id'] in done_article_ids:
            continue
        done_article_ids.add(sample['article_id'])

        article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })

        get_proper_goodnews_names(
            article, context_counter, 'context_parts_of_speech')

        for idx in article['images'].keys():
            get_caption_proper_names(article, idx, caption_counter)

    counters = {
        'caption': caption_counter,
        'context': context_counter,
    }

    with open('./data/goodnews/name_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    client = MongoClient(host=args['host'], port=27017)

    compute_nytimes(client)
    compute_goodnews(client)


def get_proper_names(section, counter):
    if 'parts_of_speech' in section:
        parts_of_speech = section['parts_of_speech']
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)


def get_proper_goodnews_names(section, counter, pos_field):
    if pos_field in section:
        parts_of_speech = section[pos_field]
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)


def get_caption_proper_names(article, idx, counter):
    if 'caption_parts_of_speech' in article:
        parts_of_speech = article['caption_parts_of_speech'][idx]
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)


if __name__ == '__main__':
    main()

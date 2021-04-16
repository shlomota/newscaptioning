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


def main():
    base_path = "/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/dbr/"
    agentsnpy = 'dbr/_agents.npy'

    config_path = "expt/nytimes/BM2/config.yaml"
    config = yaml_to_params(config_path, overrides='')

    client = pymongo.MongoClient(host='nova', port=27017)
    db = client.nytimes

    split = 'train'
    reverse = False
    full_search = False
    rand = False
    offset = False
    agent = False
    prints = False

    if 'create_agents' in sys.argv:
        free = np.arange(101)
        np.save(agentsnpy, free)
        return

    if 'train' in sys.argv:
        split = 'train'

    if 'test' in sys.argv:
        split = 'test'

    if 'r' in sys.argv:
        reverse = True

    if 'fs' in sys.argv:
        full_search = True

    if 'rand' in sys.argv:
        rand = True
        full_search = True

    if 'print' in sys.argv:
        prints = True

    offseti = [i.startswith('offset') for i in sys.argv]
    if True in offseti:
        offseti = offseti.index(True)
        offset = True
        offseti = sys.argv[offseti][len("offset"):]
        offsetend = sys.maxsize

        if "_" in offseti:
            offseti, offsetend = offseti.split("_")
            offsetend = int(offsetend)

        offseti = int(offseti)

        print('offset', offseti, offsetend)

    agenti = [i.startswith('agent') for i in sys.argv]
    if True in agenti:
        agenti = agenti.index(True)
        agent = True
        agenti = sys.argv[agenti][len("agent"):]
        if agenti == 'r':
            free = np.load(agentsnpy)
            if free:
                agenti = np.random.choice(free)
                free = free[~np.isin(free, agenti)]
                np.save(agentsnpy, free)
            else:
                raise Exception("no more free agent indices")

        else:
            agenti = int(agenti)

    if split not in ['train', 'test']:
        raise Exception('w0t?')

    if split == 'test':
        base_path += 'test/'

    tokenizer = Tokenizer.from_params(config.get('dataset_reader').get('tokenizer'))
    indexer_params = config.get('dataset_reader').get('token_indexers')
    token_indexer = {k: TokenIndexer.from_params(p) for k, p in indexer_params.items()}
    vocab = Vocabulary.from_params(config.get('vocabulary'))

    roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.large')
    roberta.eval()

    if agent:
        agent_ids = 3000
        ids = np.load(base_path + '_ids_missing.npy')
        ids = ids[agenti * agent_ids: (agenti + 1) * agent_ids]

    else:
        ids = np.load(base_path + '_ids.npy')

    if reverse:
        ids = ids[::-1]

    if rand:
        np.random.shuffle(ids)

    if offset:
        ids = ids[offseti:offsetend]

    # ['_id', 'web_url', 'snippet', 'lead_paragraph', 'abstract', 'print_page', 'blog', 'source', 'multimedia', 'headline', \
    # 'keywords', 'pub_date', 'document_type', 'news_desk', 'section_name', 'subsection_name', 'byline',\
    # 'type_of_material', 'word_count', 'slideshow_credits', 'scraped', 'parsed', 'error', 'image_positions',\
    # 'parsed_section', 'n_images', 'language', 'parts_of_speech', 'n_images_with_faces', 'detected_face_positions', 'split']

    projection = ['_id', 'parsed_section']

    sfrom = True
    l = [os.path.basename(i) for i in glob(base_path + "*")]

    if prints:
        ids = tqdm(ids)

    for aid in ids:
        if not agent:
            if full_search:
                l = [os.path.basename(id) for id in glob(base_path + "*[!m]")]
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
        r = roberta.extract_features(context['roberta']).detach()

        torch.save(r, base_path + aid)


if __name__ == '__main__':
    main()

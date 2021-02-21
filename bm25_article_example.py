import os
from PIL import Image
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
import pandas as pd

# Assume that you've already restored the database and the mongo server is running
# client = MongoClient(host='localhost', port=27017)
client = MongoClient(host='132.67.248.168', port=27017)

# All of our NYTimes800k articles sit in the database `nytimes`
db = client.nytimes

# Here we select a random article in the training set.
article = db.articles.find_one({'split': 'train'})

# You can visit the original web page where this article came from
url = article['web_url']

# Each article contains a lot of fields. If you want the title, then
title = article['headline']['main'].strip()

# If you want the article text, then you will need to manually merge all
# paragraphs together.
sections = article['parsed_section']
paragraphs = []
for section in sections:
    if section['type'] == 'paragraph':
        paragraphs.append(section['text'])


# To get the caption of the first image in the article
pos = article['image_positions'][0]
caption = sections[pos]['text'].strip()

# If you want to load the actual image into memory
image_dir = '/home/joberant/nlp_fall_2021/shlomotannor/transform-and-tell/data/nytimes/images_processed' # change this accordingly
image_path = os.path.join(image_dir, f"{sections[pos]['hash']}.jpg")
image = Image.open(image_path)

# # You can also load the pre-computed FaceNet embeddings of the faces in the image
# facenet_embeds = sections[pos]['facenet_details']['embeddings']
#
# # Object embeddings are stored in a separate collection due to a size limit in mongo
# obj = db.objects.find_one({'_id': sections[pos]['hash']})
# object_embeds = obj['object_features']

tokenized_corpus = [doc.split(" ") for doc in paragraphs]
bm25 = BM25Okapi(tokenized_corpus)
query = caption
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

df = pd.DataFrame(columns=["text", "score"])
df.text = paragraphs
df.score = doc_scores
df.to_csv("result.csv")
a = 5

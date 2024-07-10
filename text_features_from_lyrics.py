import json

with open(r'data.json', 'r') as file:
    raw_data = json.load(file)

lyrics = raw_data[0]['lyrics']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(lyrics, num_topics, num_words):
    vectorizer = CountVectorizer(input='content', strip_accents=None, stop_words='english')
    dtm = vectorizer.fit_transform(lyrics)
    lda = LatentDirichletAllocation(n_components=num_topics)
    lda.fit(dtm)

    topics = []
    for _, topic in enumerate(lda.components_):
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        topics.append(words)

    new_topics = set(tuple(topic) for topic in topics)
    new_topics = [list(new_topic) for new_topic in new_topics]
    return new_topics

from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(lyrics):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(lyrics)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


def ngram_frequency(lyrics, n=7):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngram_matrix = vectorizer.fit_transform(lyrics)
    feature_names = vectorizer.get_feature_names_out()
    return ngram_matrix, feature_names

import time
tic = time.time()
topics = extract_topics(lyrics=[lyrics], num_topics=3, num_words=3)
_, features = compute_tfidf([lyrics])
_, freq_words = ngram_frequency([lyrics])
toc = time.time()

print(topics, "\n\n", features, "\n\n", freq_words)
print("\n", toc-tic)
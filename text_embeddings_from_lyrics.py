from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel
import numpy as np

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

def generate_embeddings(model, words):
    return model.encode(words)

def main(lyrics):
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    embeddings = []

    topics = extract_topics([lyrics], num_topics=3, num_words=3)
    for topic in topics:
        topic_embeddings = generate_embeddings(model=model, words=topic)
        embeddings.append(topic_embeddings.flatten())

    _, imp_words = compute_tfidf([lyrics])
    imp_word_embeddings = generate_embeddings(model, imp_words)
    embeddings.append(imp_word_embeddings.flatten())

    _, freq_words = ngram_frequency(lyrics=[lyrics], n=7)
    freq_words_embedding = generate_embeddings(model, freq_words)
    embeddings.append(freq_words_embedding.flatten())

    print(embeddings, "\n\n")

    max_length = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = [np.pad(embedding, (0, max_length - embedding.shape[0])) for embedding in embeddings]

    final_embeddings_2d = np.vstack(padded_embeddings[:3])

    return final_embeddings_2d

if __name__ == '__main__':
    import json
    with open('data.json', 'r') as file:
        raw_data = json.load(file)
    lyrics = raw_data[0]['lyrics']
    print(main(lyrics).shape)
    print("\n\n")
    print(main(lyrics))

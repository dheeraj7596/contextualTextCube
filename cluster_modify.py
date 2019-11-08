from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import KMeans
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pickle
import numpy as np
import flair, torch

flair.device = torch.device('cpu')


def get_tok_vecs(word, df):
    tok_vecs = []
    for index, row in df.iterrows():
        line = row["sentence"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            sentence = Sentence(sent)
            for token_ind, token in enumerate(sentence):
                if token.text == word:
                    embedding.embed(sentence)
                    tok_vec = token.embedding.cpu().numpy()
                    tok_vecs.append(tok_vec)
                    break
    return tok_vecs


def cluster_analyse(word, df, threshold=0.7):
    km = KMeans(n_clusters=2, n_jobs=-1)
    tok_vecs = get_tok_vecs(word, df)
    km.fit(tok_vecs)
    cc = km.cluster_centers_
    sim = cosine_similarity(cc[0].reshape(1, -1), cc[1].reshape(1, -1))[0][0]
    if sim > threshold:
        return [np.mean(tok_vecs, axis=0)]
    else:
        return cc


def get_cluster(tok_vec, cc):
    sim0 = cosine_similarity(cc[0].reshape(1, -1), tok_vec.reshape(1, -1))[0][0]
    sim1 = cosine_similarity(cc[1].reshape(1, -1), tok_vec.reshape(1, -1))[0][0]
    if sim0 > sim1:
        return 0
    else:
        return 1


def to_tokenized_string(sentence):
    tokenized = " ".join([t.text for t in sentence.tokens])
    return tokenized


def make_word_cluster(df, embedding):
    stop_words = set(stopwords.words('english'))
    word_cluster = {}
    for index, row in df.iterrows():
        line = row["sentence"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            sentence = Sentence(sent)
            embedding.embed(sentence)
            for token_ind, token in enumerate(sentence):
                word = token.text
                if word in stop_words:
                    continue
                if word not in word_cluster:
                    cc = cluster_analyse(word, df)
                    if len(cc) == 2:
                        tok_vec = token.embedding.cpu().numpy()
                        word_cluster[word] = cc
                        cluster = get_cluster(tok_vec, cc)
                        sentence.tokens[token_ind].text = word + "$" + str(cluster)
                    else:
                        word_cluster[word] = cc
                else:
                    if len(word_cluster[word]) == 2:
                        tok_vec = token.embedding.cpu().numpy()
                        cluster = get_cluster(tok_vec, word_cluster[word])
                        sentence.tokens[token_ind].text = word + "$" + str(cluster)
            sentences[sentence_ind] = to_tokenized_string(sentence)
        df["sentence"][index] = " ".join(sentences)
    return df, word_cluster


def make_word_vec(word_cluster):
    word_vec = {}
    for word in word_cluster:
        cc = word_cluster[word]
        if len(cc) == 2:
            word1 = word + "$0"
            word_vec[word1] = cc[0]
            word2 = word + "$1"
            word_vec[word2] = cc[1]
        else:
            word_vec[word] = cc[0]
    return word_vec


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "./data/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))

    df, word_cluster = make_word_cluster(df, embedding)

    pickle.dump(df, open(pkl_dump_dir + "df_contextualized.pkl", "wb"))

    word_vec = make_word_vec(word_cluster)
    pickle.dump(word_vec, open(pkl_dump_dir + "word_2_bertvec.pkl", "wb"))
    pickle.dump(word_cluster, open(pkl_dump_dir + "word_cluster_center.pkl", "wb"))

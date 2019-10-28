from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import KMeans
from nltk import sent_tokenize
import pickle


def get_sentences(dataset):
    path = "./data/" + dataset + "/dataset.txt"
    f = open(path, "r")
    lines = f.readlines()
    sentences = []
    for line in lines:
        line = line.strip()
        sentences.extend(sent_tokenize(line))
    f.close()
    return sentences


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "./data/"
    dataset = "nyt"
    sentences = ["An apple a day keeps the doctor away.",
                 "I eat apple daily",
                 "Apple is an American multinational technology company headquartered in Cupertino, California"]
    km = KMeans(n_clusters=2)
    tok_vecs = []
    mapping = {}
    i = 0

    for sentence_ind, sent in enumerate(sentences):
        sentence = Sentence(sent)
        embedding.embed(sentence)
        for token_ind, token in enumerate(sentence):
            tok_vecs.append(token.embedding.detach().numpy())
            mapping[i] = {"token": token_ind, "sentence": sentence_ind}
            i += 1

    km.fit(tok_vecs)
    labels = km.labels_
    for i, label in enumerate(labels):
        sentence = sentences[mapping[i]["sentence"]]
        sent = Sentence(sentence)
        token_ind = mapping[i]["token"]
        print("Sentence: ", sent)
        print("Word: ", sent.tokens[token_ind])
        print("Substring: ", sent.tokens[token_ind - 2:token_ind + 2])
        print("label: ", label)
        print("*" * 20)
    print(km.labels_)

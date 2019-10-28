from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import KMeans
from nltk import sent_tokenize
import pickle
import flair, torch

flair.device = torch.device('cuda:2')


def get_sentences(path):
    f = open(path, "r")
    lines = f.readlines()
    sentences = []
    for line in lines:
        line = line.strip()
        sentences.extend(sent_tokenize(line))
    f.close()
    return sentences


def print_results(km, sentences, mapping):
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


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    dump_dir = basepath + dataset
    path = basepath + dataset + "dataset.txt"
    pkl_dump_dir = basepath + dataset

    sentences = get_sentences(path)
    km = KMeans(n_clusters=10)
    tok_vecs = []
    mapping = {}
    i = 0
    except_counter = 0

    print("Getting embeddings..")
    for sentence_ind, sent in enumerate(sentences):
        sentence = Sentence(sent)
        try:
            embedding.embed(sentence)
        except:
            except_counter += 1
            continue
        for token_ind, token in enumerate(sentence):
            tok_vecs.append(token.embedding.cpu().numpy())
            mapping[i] = {"token": token_ind, "sentence": sentence_ind}
            i += 1

    print("Fitting KMeans..")
    km.fit(tok_vecs)

    pickle.dump(km, open(pkl_dump_dir + "km.pkl", "wb"))
    pickle.dump(mapping, open(pkl_dump_dir + "mapping.pkl", "wb"))

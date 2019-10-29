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
    sentences = sentences[:25000]
    km = KMeans(n_clusters=10, n_jobs=-1)
    tok_vecs = []
    mapping = {}
    i = 0
    except_counter = 0
    token_length_exceed_counter = 0

    print("Getting embeddings..")
    processed_sentences = []
    for sentence_ind, sent in enumerate(sentences):
        if sentence_ind % 10000 == 0:
            print("Finished sentences: " + str(sentence_ind) + " out of " + str(len(sentences)))
        sentence = Sentence(sent)
        if len(sentence.tokens) > 200:
            token_length_exceed_counter += 1
            print("Token length exceeded for : " + str(sentence_ind) + " Token exceed counter: " + str(token_length_exceed_counter))
            continue
        try:
            embedding.embed(sentence)
        except Exception as e:
            except_counter += 1
            print("Exception Counter: ", except_counter, sentence_ind, e)
            continue
        processed_sentences.append(sent)
        for token_ind, token in enumerate(sentence):
            tok_vecs.append(token.embedding.cpu().numpy())
            mapping[i] = {"token": token_ind, "sentence": sentence_ind}
            i += 1

    print("Fitting KMeans..")
    km.fit(tok_vecs)

    pickle.dump(km, open(pkl_dump_dir + "km.pkl", "wb"))
    pickle.dump(processed_sentences, open(pkl_dump_dir + "processed_sentences.pkl", "wb"))
    pickle.dump(mapping, open(pkl_dump_dir + "mapping.pkl", "wb"))

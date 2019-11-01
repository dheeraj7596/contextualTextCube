from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import KMeans
from analyse import get_word_context
from nltk import sent_tokenize
import pickle
import flair, torch
import os
import sys

flair.device = torch.device('cuda:2')


class Word():
    def __init__(self, name, context, tok_vec, label, cluster=None):
        self.name = name
        self.context = context
        self.tok_vec = tok_vec
        self.label = label
        self.cluster = cluster


def get_word_sentences(word, dataset):
    basepath = "/data3/jingbo/dheeraj/"
    path = basepath + dataset + "dataset.txt"
    label_path = basepath + dataset + "labels.txt"

    f = open(path, "r")
    f1 = open(label_path, "r")

    lines = f.readlines()
    label_lines = f1.readlines()

    sent_list = []
    label_list = []

    for i, line in enumerate(lines):
        line = line.strip().lower()
        label_line = label_lines[i].strip()
        sentences = sent_tokenize(line)
        for sent in sentences:
            if word in sent:
                sent_list.append(sent)
                label_list.append(label_line)
    f.close()
    f1.close()
    return sent_list, label_list


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"

    word = sys.argv[1]
    pkl_dump_dir = basepath + dataset + word
    os.makedirs(pkl_dump_dir, exist_ok=True)

    sentences, labels = get_word_sentences(word, dataset)
    km = KMeans(n_clusters=2, n_jobs=-1)

    tok_vecs = []
    word_obj_list = []

    except_counter = 0
    token_length_exceed_counter = 0

    print("Getting embeddings..")
    for sentence_ind, sent in enumerate(sentences):
        if sentence_ind % 1000 == 0:
            print("Finished sentences: " + str(sentence_ind) + " out of " + str(len(sentences)))
        sentence = Sentence(sent)
        if len(sentence.tokens) > 200:
            token_length_exceed_counter += 1
            print("Token length exceeded for : " + str(sentence_ind) + " Token exceed counter: " + str(
                token_length_exceed_counter))
            continue
        try:
            embedding.embed(sentence)
        except Exception as e:
            except_counter += 1
            print("Exception Counter: ", except_counter, sentence_ind, e)
            continue
        for token_ind, token in enumerate(sentence):
            if token.text != word:
                continue
            vec = token.embedding.cpu().numpy()
            word_obj = Word(name=token.text,
                            context=get_word_context(token_ind, sent),
                            tok_vec=vec,
                            label=labels[sentence_ind]
                            )
            tok_vecs.append(vec)
            word_obj_list.append(word_obj)

    print("Fitting KMeans on " + str(len(tok_vecs)) + " tokens..")
    km.fit(tok_vecs)

    pickle.dump(km, open(pkl_dump_dir + "km.pkl", "wb"))
    pickle.dump(sentences, open(pkl_dump_dir + "sentences.pkl", "wb"))
    pickle.dump(word_obj_list, open(pkl_dump_dir + "word_obj_list.pkl", "wb"))

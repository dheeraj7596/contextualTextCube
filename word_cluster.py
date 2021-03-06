from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import KMeans
from analyse import get_word_context
from Word import Word
from nltk import sent_tokenize
from visualise import visualise
import pickle
import flair, torch
import os
import sys

flair.device = torch.device('cuda:2')


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
    print("WORD: ", word)
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

    print("Visualising..")
    f = visualise(tok_vecs, km.labels_, None, pkl_dump_dir + "/plot.png")

    for i, label in enumerate(km.labels_):
        word_obj_list[i].cluster = label

    f0 = open(pkl_dump_dir + "/0.txt", "w")
    f1 = open(pkl_dump_dir + "/1.txt", "w")
    for word in word_obj_list:
        if word.cluster == 0:
            f0.write(word.context[1])
            f0.write("\n")
        else:
            f1.write(word.context[1])
            f1.write("\n")
    f0.close()
    f1.close()

    print("Dumping pickles..")
    pickle.dump(km, open(pkl_dump_dir + "/km.pkl", "wb"))
    pickle.dump(sentences, open(pkl_dump_dir + "/sentences.pkl", "wb"))
    pickle.dump(word_obj_list, open(pkl_dump_dir + "/word_obj_list.pkl", "wb"))

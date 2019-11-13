from cocube_utils import get_distinct_labels, train_classifier
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pickle
import numpy as np


def create_index(word_vec):
    word_to_index = {}
    index_to_word = {}
    words = list(word_vec.keys())
    for i, word in enumerate(words):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word


def get_label_term_dict(labels, word_vec):
    label_term_dict = defaultdict(set)
    for i in labels:
        terms = i.split("_")
        if i == "stocks_and_bonds":
            terms = ["stocks", "bonds"]
        elif i == "the_affordable_care_act":
            terms = ["affordable", "care", "act"]

        for term in terms:
            try:
                temp = word_vec[term]
                label_term_dict[i].add(term)
            except:
                pass
            try:
                temp = word_vec[term + "$0"]
                label_term_dict[i].add(term + "$0")
            except:
                pass
            try:
                temp = word_vec[term + "$1"]
                label_term_dict[i].add(term + "$1")
            except:
                pass

    return label_term_dict


def get_inv_doc_freq(df):
    docfreq = {}
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    N = len(df)
    for word in docfreq:
        docfreq[word] = np.log(N / docfreq[word])
    return docfreq


def update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index,
                           index_to_word, inv_docfreq):
    label_docs_dict = {}
    label_count = len(label_to_index)
    term_count = len(word_to_index)

    E_LT = np.zeros((label_count, term_count))

    for l in label_term_dict:
        label_docs_dict[l] = []

    for index, row in df.iterrows():
        line = row["sentence"]
        label_docs_dict[pred_labels[index]].append(line)

    for l in label_docs_dict:
        docs = label_docs_dict[l]
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            E_LT[label_to_index[l]][word_to_index[name]] = rel_freq[i] * inv_docfreq[name]

    word_map = {}
    for l in range(label_count):
        inds = E_LT[l].argsort()[::-1][:10]
        for word_ind in inds:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if E_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], E_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], E_LT[l][word_ind])
    for word in word_map:
        label, val = word_map[word]
        label_term_dict[label].add(word)
    return label_term_dict


def print_label_term_dict(label_term_dict):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            print(val)


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_contextualized_clean_removed_stopwords.pkl", "rb"))
    word_vec = pickle.load(open(pkl_dump_dir + "word_vec_clean_removed_stopwords.pkl", "rb"))

    word_to_index, index_to_word = create_index(word_vec)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_dict(labels, word_vec)
    inv_docfreq = get_inv_doc_freq(df)

    t = 5

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")
        pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, i)
        print("Updating label term dict..")
        label_term_dict = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label,
                                                 word_to_index, index_to_word, inv_docfreq)
        print_label_term_dict(label_term_dict)
        print("#" * 80)

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
        if i == "stocks_and_bonds":
            terms = ["stocks", "bonds$1"]
            label_term_dict[i] = set(terms)
        elif i == "the_affordable_care_act":
            terms = ["affordable$0 care$0"]
            label_term_dict[i] = set(terms)
        elif i == "gun_control":
            terms = ["gun control"]
            label_term_dict[i] = set(terms)
        elif i == "federal_budget":
            terms = ["tax", "debt"]
            label_term_dict[i] = set(terms)
        elif i == "energy_companies":
            terms = ["energy", "solar"]
            label_term_dict[i] = set(terms)
        elif i == "cosmos":
            terms = ["space$0", "nasa", "planets"]
            label_term_dict[i] = set(terms)
        elif i == "gay_rights":
            terms = ["gay rights"]
            label_term_dict[i] = set(terms)
        elif i == "international_business":
            terms = ["bank", "european", "euro"]
            label_term_dict[i] = set(terms)
        elif i == "law_enforcement":
            terms = ["law$1 enforcement"]
            label_term_dict[i] = set(terms)
        else:
            terms = i.split("_")
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
                           index_to_word, inv_docfreq, it):
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
        n = 10 * (it + 1)
        inds = E_LT[l].argsort()[::-1][:n]
        for word_ind in inds:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if E_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], E_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], E_LT[l][word_ind])
    label_term_dict = defaultdict(set)
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


def get_label_term_json(pkl_dump_dir):
    import json
    dic = json.load(open(pkl_dump_dir + "seedwords.json", "r"))
    return dic


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_contextualized_clean_removed_stopwords.pkl", "rb"))
    word_vec = pickle.load(open(pkl_dump_dir + "word_vec_clean_removed_stopwords.pkl", "rb"))

    word_to_index, index_to_word = create_index(word_vec)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(pkl_dump_dir)
    # label_term_dict = get_label_term_dict(labels, word_vec)
    inv_docfreq = get_inv_doc_freq(df)

    t = 5

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")
        pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label)
        print("Updating label term dict..")
        label_term_dict = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label,
                                                 word_to_index, index_to_word, inv_docfreq, i)
        print_label_term_dict(label_term_dict)
        print("#" * 80)

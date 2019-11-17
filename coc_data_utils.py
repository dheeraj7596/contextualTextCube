import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def get_label_term_json(pkl_dump_dir):
    import json
    dic = json.load(open(pkl_dump_dir + "seedwords.json", "r"))
    return dic


def create_index(word_vec):
    word_to_index = {}
    index_to_word = {}
    words = list(word_vec.keys())
    for i, word in enumerate(words):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word


def print_label_term_dict(label_term_dict):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            print(val)


def get_term_freq(df):
    term_freq = defaultdict(int)
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for w in words:
            term_freq[w] += 1
    return term_freq


def get_doc_freq(df):
    docfreq = {}
    docfreq["UNK"] = len(df)
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def get_inv_doc_freq(df, docfreq):
    N = len(df)
    for word in docfreq:
        docfreq[word] = np.log(N / docfreq[word])
    return docfreq


def get_label_docs_dict(df, label_term_dict, pred_labels):
    label_docs_dict = {}
    for l in label_term_dict:
        label_docs_dict[l] = []
    for index, row in df.iterrows():
        line = row["sentence"]
        label_docs_dict[pred_labels[index]].append(line)
    return label_docs_dict


def plot_hist(values, bins):
    plt.figure()
    n = plt.hist(values, color='blue', edgecolor='black', bins=bins)
    print(n)
    plt.show()


def modify_df(df, docfreq, threshold=5):
    UNK = "UNK"
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for i, w in enumerate(words):
            if docfreq[w] < threshold:
                words[i] = UNK
        df["sentence"][index] = " ".join(words)
    return df

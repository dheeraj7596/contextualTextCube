from cocube_utils import get_distinct_labels, train_classifier
from coc_data_utils import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pickle
import numpy as np
import sys
import math
import copy


def get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                       term_count, word_to_index, doc_freq_thresh):
    E_LT = np.zeros((label_count, term_count))
    components = {}
    for l in label_docs_dict:
        components[l] = {}
        docs = label_docs_dict[l]
        docfreq_local = calculate_doc_freq(docs)
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq_local[name] < doc_freq_thresh:
                    continue
            except:
                continue
            E_LT[label_to_index[l]][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[name] \
                                                           * np.tanh(rel_freq[i])
            components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                   "idf": inv_docfreq[name],
                                   "rel_freq": np.tanh(rel_freq[i]),
                                   "rank": E_LT[label_to_index[l]][word_to_index[name]]}
    return E_LT, components


def update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict):
    word_map = {}
    for l in range(label_count):
        if not np.any(E_LT):
            n = 0
        else:
            n = min(n1 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_popular = E_LT[l].argsort()[::-1][:n]

        if not np.any(F_LT):
            n = 0
        else:
            n = min(n2 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_exclusive = F_LT[l].argsort()[::-1][:n]

        for word_ind in inds_popular:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if E_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], E_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], E_LT[l][word_ind])

        for word_ind in inds_exclusive:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if F_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], F_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], F_LT[l][word_ind])

    label_term_dict = defaultdict(set)
    for word in word_map:
        label, val = word_map[word]
        label_term_dict[label].add(word)
    return label_term_dict


def update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index,
                           index_to_word, inv_docfreq, docfreq, it, n1, n2, doc_freq_thresh=5):
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)
    E_LT, components = get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict,
                                          label_to_index, term_count, word_to_index, doc_freq_thresh)
    F_LT = np.zeros((label_count, term_count))

    label_term_dict = update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict)
    return label_term_dict, components


def create_index_maps(df):
    word_to_index = {}
    index_to_word = {}
    sentences = df.sentence
    word_set = set()
    for sent in sentences:
        word_set.update(set(sent.strip().split()))

    for i, word in enumerate(word_set):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word


def disambiguate_seeds(df, labels, label_term_dict, label_to_index, index_to_label, word_to_index, index_to_word,
                       inv_docfreq, docfreq, word_interpretation):
    all_interpretations_seeds = {}
    for l in label_term_dict:
        for word in label_term_dict[l]:
            all_interpretations_seeds[word] = []
            if word in word_interpretation:
                for interp in word_interpretation[word]:
                    temp_word = word + "$" + str(word_interpretation[word][interp])
                    all_interpretations_seeds[word].append(temp_word)
            else:
                all_interpretations_seeds[word] = [word]

    new_label_term_dict = {}
    for l in label_term_dict:
        new_label_term_dict[l] = []
        for word in label_term_dict[l]:
            new_label_term_dict[l] += all_interpretations_seeds[word]

    pred_labels = train_classifier(df, labels, new_label_term_dict, label_to_index, index_to_label)
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)
    E_LT, components = get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict,
                                          label_to_index, term_count, word_to_index, 7)

    final_label_term_dict = {}
    for l in label_term_dict:
        final_label_term_dict[l] = []
        for word in label_term_dict[l]:
            interp_words = all_interpretations_seeds[word]
            values = []
            for w in interp_words:
                values.append(E_LT[label_to_index[l]][word_to_index[w]])
            right_interp = interp_words[values.index(max(values))]
            final_label_term_dict[l].append(right_interp)

    return final_label_term_dict


if __name__ == "__main__":
    pre_trained = 0
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_tokenized_contextualized_lesk_clean_child.pkl", "rb"))
    word_interpretation = pickle.load(open(pkl_dump_dir + "word_interpretation_lesk.pkl", "rb"))

    word_to_index, index_to_word = create_index_maps(df)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_child.json")

    docfreq = get_doc_freq(df)
    inv_docfreq = get_inv_doc_freq(df, docfreq)

    # df = modify_df(df, docfreq, 5)

    label_term_dict = disambiguate_seeds(df, labels, label_term_dict, label_to_index, index_to_label, word_to_index,
                                         index_to_word, inv_docfreq, docfreq, word_interpretation)

    print("After Disambiguating seeds..")
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            try:
                print(val)
            except Exception as e:
                print("Exception occured: ", e, val)

    t = 10

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")
        if i == 0 and pre_trained == 1:
            pred_labels = pickle.load(open(pkl_dump_dir + "seedwords_pred.pkl", "rb"))
        else:
            pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label)
        # if i == 0:
        #     pickle.dump(pred_labels, open(pkl_dump_dir + "seedwords_pred.pkl", "wb"))
        print("Updating label term dict..")
        label_term_dict, components = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index,
                                                             index_to_label, word_to_index, index_to_word, inv_docfreq,
                                                             docfreq, i, n1=7, n2=7)
        print_label_term_dict(label_term_dict, components)
        print("#" * 80)

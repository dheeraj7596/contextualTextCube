from cocube_utils import get_distinct_labels, train_classifier
from coc_data_utils import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pickle
import numpy as np
import sys
import copy


def get_popular_matrix(index_to_word, inv_docfreq, label_count, label_docs_dict, label_to_index,
                       term_count, word_to_index):
    E_LT = np.zeros((label_count, term_count))
    for l in label_docs_dict:
        docs = label_docs_dict[l]
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            E_LT[label_to_index[l]][word_to_index[name]] = np.tanh(rel_freq[i])
    for l in range(label_count):
        for t in range(term_count):
            E_LT[l][t] = E_LT[l][t] * inv_docfreq[index_to_word[t]]
    return E_LT


def get_exclusive_matrix(doc_freq_thresh, index_to_label, index_to_word, inv_docfreq, label_count, label_docs_dict,
                         label_to_index, term_count, word_to_index):
    E_LT = np.zeros((label_count, term_count))
    components = {}
    for l in label_docs_dict:
        components[l] = {}
    for l in label_docs_dict:
        docs = label_docs_dict[l]
        docfreq = calculate_doc_freq(docs)
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        freq = np.sum(X_arr, axis=0)
        rel_freq = freq / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq[name] < doc_freq_thresh:
                    continue
            except:
                continue
            E_LT[label_to_index[l]][word_to_index[name]] = (rel_freq[i] ** 0.2) * (freq[i] ** 1.5)
            components[l][name] = {"relfreq": rel_freq[i] ** 0.2, "freq": freq[i] ** 1.5}
    for l in range(label_count):
        zero_counter = 0
        for t in range(term_count):
            flag = 0
            if E_LT[l][t] == 0:
                continue
            col_list = list(E_LT[:, t])
            temp_list = copy.deepcopy(col_list)
            temp_list.pop(l)
            den = np.nanmax(temp_list)
            if den == 0:
                flag = 1
                den = 0.0001
                zero_counter += 1
            temp = E_LT[l][t] / (den ** 0.2)
            E_LT[l][t] = temp * inv_docfreq[index_to_word[t]]
            components[index_to_label[l]][index_to_word[t]]["ratio"] = components[index_to_label[l]][index_to_word[t]][
                                                                           "relfreq"] / (den ** 0.2)
            components[index_to_label[l]][index_to_word[t]]["idf"] = inv_docfreq[index_to_word[t]]
            components[index_to_label[l]][index_to_word[t]]["rare"] = flag
            components[index_to_label[l]][index_to_word[t]]["rank"] = E_LT[l][t]
        print(index_to_label[l], zero_counter)
    return E_LT, components


def update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict):
    word_map = {}
    for l in range(label_count):
        if not np.any(E_LT):
            n = 0
        else:
            n = min(n1 * (it + 1), int(np.log(label_docs_dict[index_to_label[l]])))
        inds_popular = E_LT[l].argsort()[::-1][:n]

        if not np.any(F_LT):
            n = 0
        else:
            n = min(n2 * (it + 1), int(np.log(label_docs_dict[index_to_label[l]])))
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
                           index_to_word, inv_docfreq, it, n1, n2, doc_freq_thresh=5, flag=1):
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

    if flag == 0:
        E_LT = np.zeros((label_count, term_count))
        F_LT, components = get_exclusive_matrix(doc_freq_thresh, index_to_label, index_to_word, inv_docfreq,
                                                label_count, label_docs_dict, label_to_index, term_count, word_to_index)
    elif flag == 1:
        E_LT = get_popular_matrix(index_to_word, inv_docfreq, label_count, label_docs_dict, label_to_index, term_count,
                                  word_to_index)
        F_LT, components = get_exclusive_matrix(doc_freq_thresh, index_to_label, index_to_word, inv_docfreq,
                                                label_count, label_docs_dict, label_to_index, term_count, word_to_index)
    else:
        E_LT = get_popular_matrix(index_to_word, inv_docfreq, label_count, label_docs_dict, label_to_index, term_count,
                                  word_to_index)
        F_LT = np.zeros((label_count, term_count))

    label_term_dict = update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict)
    return label_term_dict, components


if __name__ == "__main__":
    pre_trained = int(sys.argv[1])
    flag = int(sys.argv[2])
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_tokenized_contextualized_clean_removed_stopwords.pkl", "rb"))
    word_vec = pickle.load(open(pkl_dump_dir + "word_vec_tokenized_clean_removed_stopwords.pkl", "rb"))

    word_to_index, index_to_word = create_index(word_vec)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(pkl_dump_dir)

    docfreq = get_doc_freq(df)
    inv_docfreq = get_inv_doc_freq(df, docfreq)

    # df = modify_df(df, docfreq, 5)
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
                                                             i, n1=5, n2=7, flag=flag)
        print_label_term_dict(label_term_dict, components)
        print("#" * 80)

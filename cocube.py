from cocube_utils import get_distinct_labels, train_classifier
from coc_data_utils import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pickle
import numpy as np
import copy


def update(E_LT, index_to_label, index_to_word, it, label_count):
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


def update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index,
                           index_to_word, inv_docfreq, it):
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

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
        zero_counter = 0
        for t in range(term_count):
            if E_LT[l][t] == 0:
                continue
            col_list = list(E_LT[:, t])
            temp_list = copy.deepcopy(col_list)
            temp_list.pop(l)
            den = np.nanmax(temp_list)
            if den == 0:
                den = 1
                zero_counter += 1
            temp = E_LT[l][t]
            # temp = E_LT[l][t] / den
            E_LT[l][t] = temp * inv_docfreq[index_to_word[t]]
        print(index_to_label[l], zero_counter)

    label_term_dict = update(E_LT, index_to_label, index_to_word, it, label_count)
    return label_term_dict


if __name__ == "__main__":
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
        if i == 0:
            pred_labels = pickle.load(open(pkl_dump_dir + "seedwords_pred.pkl", "rb"))
        else:
            pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label)
        # if i == 0:
        #     pickle.dump(pred_labels, open(pkl_dump_dir + "seedwords_pred.pkl", "wb"))
        print("Updating label term dict..")
        label_term_dict = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label,
                                                 word_to_index, index_to_word, inv_docfreq, i)
        print_label_term_dict(label_term_dict)
        print("#" * 80)

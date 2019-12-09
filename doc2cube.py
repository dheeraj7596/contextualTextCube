from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from scipy.special import softmax
from scipy.stats import entropy
from gensim.models import Word2Vec
import numpy as np
import pickle


def get_distinct_labels(df):
    labels = list(set(df["label"]))
    label_to_index = {}
    index_to_label = {}

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label

    return labels, label_to_index, index_to_label


def get_doc_freq(df):
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
    return docfreq


def get_UD(df, word_vec):
    dim = list(word_vec.values())[0].shape[0]
    num_docs = len(df)
    U_D = np.zeros((num_docs, dim))
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for word in words:
            U_D[index] += word_vec[word]
    return U_D


def get_UL(A_LT, U_T):
    return A_LT


def get_UT(word_vec):
    word_to_index = {}
    index_to_word = {}
    U_T = []

    words = list(word_vec.wv.vocab.keys())
    for i, word in enumerate(words):
        word_to_index[word] = i
        index_to_word[i] = word
        U_T.append(word_vec[word])

    return word_to_index, index_to_word, U_T


def get_ATD(df, word_to_index):
    A_TD = np.zeros((len(word_to_index), len(df)))
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for word in words:
            A_TD[word_to_index[word]][index] = 1
    return A_TD


def get_ALT(index_to_label, word_to_index):
    num_labels = len(index_to_label)
    num_words = len(word_to_index)
    A_LT = np.zeros((num_labels, num_words))
    for l in index_to_label:
        label = index_to_label[l]

        if label == "the_affordable_care_act":
            A_LT[l][word_to_index["affordable$0"]] = 1
            A_LT[l][word_to_index["care$0"]] = 1
        elif label == "stocks_and_bonds":
            A_LT[l][word_to_index["stocks"]] = 1
            A_LT[l][word_to_index["bonds$1"]] = 1
        elif label == "gun_control":
            A_LT[l][word_to_index["gun"]] = 1
        elif label == "federal_budget":
            A_LT[l][word_to_index["budget"]] = 1
        elif label == "energy_companies":
            A_LT[l][word_to_index["energy"]] = 1
        elif label == "cosmos":
            A_LT[l][word_to_index["cosmos$1"]] = 1
        elif label == "gay_rights":
            A_LT[l][word_to_index["gay"]] = 1
        elif label == "international_business":
            A_LT[l][word_to_index["business"]] = 1
        elif label == "law_enforcement":
            A_LT[l][word_to_index["law$0"]] = 1
        else:
            index = word_to_index[label]
            A_LT[l][index] = 1
    return A_LT


def apply_softmax(R_TL):
    rows, cols = R_TL.shape
    for i in range(rows):
        R_TL[i] = softmax(R_TL[i][:])
    return R_TL


def compute_dim_focal_score(R_TL):
    f_dim_focal_score = []
    term_count, label_count = R_TL.shape
    for i in range(term_count):
        uniform = np.array([1 / label_count] * label_count)
        temp = R_TL[i][:]
        kl = entropy(temp, uniform)
        f_dim_focal_score.append(kl / np.log(label_count))
    return f_dim_focal_score


def update_UD(U_T, A_TD, f_dim_focal_score):
    f_list = list(f_dim_focal_score)
    term_count, doc_count = A_TD.shape
    f_mat = np.tile(np.array([f_list]).transpose(), (1, doc_count))
    weight = np.multiply(A_TD, f_mat)
    return np.matmul(np.transpose(weight), U_T)


def update_ALT(A_LT, R_TL, index_to_word, num_docs, docfreq, threshold=0.8):
    term_count, label_count = R_TL.shape
    for i in range(term_count):
        for j in range(label_count):
            temp = R_TL[i][j] * (docfreq[index_to_word[i]] / num_docs)
            if temp > threshold:
                A_LT[j][i] = 1
    return A_LT


def print_A_LT(A_LT, index_to_label, index_to_word):
    label_count, term_count = A_LT.shape
    for i in range(label_count):
        print("*" * 80)
        print("For Label ", index_to_label[i] + " : ")
        for j in range(term_count):
            if A_LT[i][j]:
                print(index_to_word[j])


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_tokenized_limit_clean_parent.pkl", "rb"))

    word_vec = Word2Vec.load(pkl_dump_dir + "w2v.model_parent")
    # word_vec = pickle.load(open(pkl_dump_dir + "word_vec_tokenized_clean_removed_stopwords.pkl", "rb"))
    labels, label_to_index, index_to_label = get_distinct_labels(df)

    # U_D = get_UD(df, word_vec)
    # U_D = pickle.load(open(pkl_dump_dir + "U_D.pkl", "rb"))
    word_to_index, index_to_word, U_T = get_UT(word_vec)
    A_LT = get_ALT(index_to_label, word_to_index)
    U_L = get_UL(A_LT, U_T)
    A_TD = get_ATD(df, word_to_index)
    U_D = np.transpose(A_TD)
    docfreq = get_doc_freq(df)
    t = 20
    threshold = 0.4

    for i in range(t):
        print("ITERATION: ", i)
        print("Computing R_DL..")
        R_DL = np.matmul(U_D, np.transpose(U_L))
        print("Computing R_TL..")
        R_TL = np.matmul(A_TD, R_DL)
        R_TL = apply_softmax(R_TL)
        print("Computing Dim focal score..")
        f_dim_focal_score = compute_dim_focal_score(R_TL)
        print("Updating U_D..")
        U_D = update_UD(U_T, A_TD, f_dim_focal_score)
        print("Updating A_LT..")
        A_LT = update_ALT(A_LT, R_TL, index_to_word, len(df), docfreq, threshold)
        print_A_LT(A_LT, index_to_label, index_to_word)
        print("Updating A_L..")
        U_L = np.matmul(A_LT, U_T)
        print("*" * 80)

    preds = []
    for u in U_D:
        maxi_sim = -1
        maxi_l = None
        for i, l in enumerate(U_L):
            sim = cosine_similarity(u.reshape(1, -1), l.reshape(1, -1))[0][0]
            if sim > maxi_sim:
                maxi_sim = sim
                maxi_l = i
        preds.append(index_to_label[maxi_l])

    print(classification_report(df["label"], preds))

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def get_one_hot(label_to_index, labels, l):
    n = len(labels)
    t = np.zeros((n,))
    t[label_to_index[l]] = 1
    return t


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_tokenized_contextualized_clean_removed_stopwords.pkl", "rb"))
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["sentence"])
    y = []
    for l in df["label"]:
        y.append(label_to_index[l])

    names = vectorizer.get_feature_names()
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    pass


def ret_max(names, coeff, i):
    temp = coeff[i].argsort()[-10:][::-1]
    for i in temp:
        print(names[i])
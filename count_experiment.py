from nltk import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import operator


def create_df(dataset):
    basepath = "./data/"
    path = basepath + dataset + "dataset.txt"
    label_path = basepath + dataset + "labels.txt"
    f = open(path, "r")
    f1 = open(label_path, "r")
    lines = f.readlines()
    label_lines = f1.readlines()
    final_dict = {}
    final_dict["sentence"] = []
    final_dict["label"] = []

    for i, line in enumerate(lines):
        line = line.strip().lower()
        label_line = label_lines[i].strip()
        final_dict["sentence"].append(line)
        final_dict["label"].append(label_line)
    df = DataFrame(final_dict)
    return df


def get_distinct_labels(dataset):
    basepath = "./data/"
    label_path = basepath + dataset + "labels.txt"

    f = open(label_path, "r")
    lines = f.readlines()
    labels = set()
    label_count_dict = {}
    for line in lines:
        label = line.strip()
        labels.add(label)
        if label in label_count_dict:
            label_count_dict[label] += 1
        else:
            label_count_dict[label] = 1
    f.close()
    return labels, label_count_dict


def decide_label(count_dict):
    maxi = 0
    max_label = None
    for l in count_dict:
        min_freq = min(list(count_dict[l].values()))
        if min_freq > maxi:
            maxi = min_freq
            max_label = l
    return max_label


def get_train_data(df, labels, label_term_dict):
    y = []
    X = []
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
        sentences = sent_tokenize(line)
        count_dict = {}
        flag = 0
        for sent in sentences:
            words = word_tokenize(sent)
            for l in labels:
                int_labels = list(set(words).intersection(set(label_term_dict[l])))
                if len(int_labels) == 0:
                    continue
                for word in words:
                    if word in int_labels:
                        flag = 1
                        if l not in count_dict:
                            count_dict[l] = {}
                        if word not in count_dict[l]:
                            count_dict[l][word] = 1
                        else:
                            count_dict[l][word] += 1

        if flag:
            lbl = decide_label(count_dict)
            y.append(lbl)
            X.append(line)
    return X, y


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "nyt/"
    df = create_df(dataset)
    le = LabelEncoder()
    labels, label_count_dict = get_distinct_labels(dataset)
    label_term_dict = {}
    for i in labels:
        terms = i.split("_")
        if i == "stocks_and_bonds":
            label_term_dict[i] = ["stocks", "bonds"]
        elif i == "the_affordable_care_act":
            label_term_dict[i] = ["affordable", "care", "act"]
        else:
            label_term_dict[i] = terms

    X, y = get_train_data(df, labels, label_term_dict)
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X)
    y_train = le.fit_transform(y)

    X_test = vectorizer.transform(df["sentence"])
    y_test = le.inverse_transform(le.transform(df["label"]))

    clf = LogisticRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = le.inverse_transform(clf.predict(X_test))
    print(classification_report(y_test, y_pred))

    pass

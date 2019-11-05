from nltk import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.metrics import classification_report
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


def get_pred_true(df, labels):
    y_true = []
    y_pred = []
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
        sentences = sent_tokenize(line)
        count_dict = {}
        flag = 0
        for sent in sentences:
            words = word_tokenize(sent)
            int_labels = list(set(words).intersection(set(labels)))
            if len(int_labels) == 0:
                continue
            prev = words[0]
            for i, word in enumerate(words):
                if word in int_labels:
                    flag = 1
                    if word not in count_dict:
                        count_dict[word] = 1
                    else:
                        count_dict[word] += 1
                else:
                    if i == 0:
                        continue
                    else:
                        temp = prev + "_" + word
                        if temp in labels:
                            flag = 1
                            if temp not in count_dict:
                                count_dict[temp] = 1
                            else:
                                count_dict[temp] += 1
                prev = word

        if flag:
            y_pred.append(max(count_dict.items(), key=operator.itemgetter(1))[0])
            y_true.append(label)
    return y_true, y_pred


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "nyt/"
    df = create_df(dataset)
    le = LabelEncoder()
    labels = ["soccer", "music", "movies", "basketball", "tennis", "business", "television", "economy", "science",
              "baseball", "politics", "hockey", "football", "golf", "dance", "environment", "abortion", "cosmos",
              "surveillance", "military", "immigration", "international_business", "energy_companies",
              "law_enforcement", "gun_control", "gay_rights", "federal_budget"]
    df = df[df.label.isin(labels)]
    df = df.reset_index(drop=True)
    y_true_names, y_pred_names = get_pred_true(df, labels)

    count_dict = {}
    for name in y_true_names:
        if name not in count_dict:
            count_dict[name] = 1
        else:
            count_dict[name] += 1

    print("The number of documents that has atleast one label in text: ", len(y_true_names))
    print("Distribution of labels in the docs that have atleast one label: ", count_dict)
    print("CLASSIFICATION REPORT: ")
    print(classification_report(y_true_names, y_pred_names))

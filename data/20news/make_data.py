from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import pickle


def make_df(train, labels):
    sentences_list = []
    label_list = []
    skip_counter = 0
    for i in range(0, len(train.data)):
        if i % 1000 == 0:
            print(i)
        label = labels[train.target[i]]
        line = train.data[i]
        line = line.strip().lower()
        sentences = sent_tokenize(line)
        flag = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 200:
                skip_counter += 1
                print("Skipped due to large tokens: ", skip_counter)
                flag = 1
                break
        if flag == 0:
            sentences_list.append(line)
            label_list.append(label)
    return sentences_list, label_list


if __name__ == "__main__":
    dic = {}
    dic["sentence"] = []
    dic["label"] = []

    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')

    labels = list(train.target_names)

    sentence_train_list, labels_train_list = make_df(train, labels)
    sentence_test_list, labels_test_list = make_df(test, labels)

    dic["sentence"].extend(sentence_train_list)
    dic["sentence"].extend(sentence_test_list)
    dic["label"].extend(labels_train_list)
    dic["label"].extend(labels_test_list)
    df = pd.DataFrame(dic)
    with open("./df_tokens_limit.pkl", "wb") as handler:
        pickle.dump(df, handler)

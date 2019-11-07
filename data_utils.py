import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from nltk import tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame


def make_one_hot(y, label_to_index):
    labels = list(label_to_index.keys())
    n_classes = len(labels)
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        i = label_to_index[label]
        current[i] = 1.0
        y_new.append(current)
    y_new = np.asarray(y_new)
    return y_new


def get_from_one_hot(pred, index_to_label):
    pred_labels = np.argmax(pred, axis=-1)
    ans = []
    for l in pred_labels:
        ans.append(index_to_label[l])
    return ans


def create_train_dev(texts, labels, tokenizer, max_sentences=15, max_sentence_length=100, max_words=20000):
    data = prep_data(max_sentence_length, max_sentences, texts, tokenizer)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, y_train, X_test, y_test


def prep_data(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        sents = tokenize.sent_tokenize(text)
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data


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

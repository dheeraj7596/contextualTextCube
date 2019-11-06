from nltk import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from model import *
from data_utils import *


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
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_count_dict, label_to_index, index_to_label


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
    glove_dir = basepath + "glove.6B"
    model_name = "count_exp"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100
    batch_size = 290

    df = create_df(dataset)
    le = LabelEncoder()
    labels, label_count_dict, label_to_index, index_to_label = get_distinct_labels(dataset)
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

    y_one_hot = make_one_hot(y, label_to_index)

    print("Fitting tokenizer...")
    tokenizer = fit_get_tokenizer(X, max_words)
    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                      max_sentences=max_sentences,
                                                      max_sentence_length=max_sentence_length,
                                                      max_words=max_words)

    print("Creating Embedding matrix...")
    embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)

    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)

    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=2, batch_size=100, callbacks=[es, mc])

    X_all = np.vstack((X_train, X_val))
    y_all = np.vstack((y_train, y_val))
    pred = model.predict(X_all)
    print("****************** CLASSIFICATION REPORT ********************")
    pred_labels = get_from_one_hot(pred, index_to_label)
    true_labels = get_from_one_hot(y_all, index_to_label)
    print(classification_report(true_labels, pred_labels))

    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")

from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from scipy.special import softmax
from keras.losses import kullback_leibler_divergence
from model import *
from data_utils import *
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_training_df(X, y, y_true):
    dic = {}
    dic["Data"] = X
    dic["Training label"] = y
    dic["True label"] = y_true
    df_X = DataFrame(dic)
    return df_X


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def argmax_label(count_dict):
    maxi = 0
    max_label = None
    for l in count_dict:
        count = 0
        for t in count_dict[l]:
            count += count_dict[l][t]
        if count > maxi:
            maxi = count
            max_label = l

    return max_label


def softmax_label(count_dict, label_to_index):
    temp = [0] * len(label_to_index)
    for l in count_dict:
        min_freq = min(list(count_dict[l].values()))
        temp[label_to_index[l]] = min_freq
    return softmax(temp)


def add_phrase_keys(count_dict, l, seed_phrases):
    if len(seed_phrases) == 0:
        return count_dict

    for ph in seed_phrases:
        words = ph.split()
        int_words = set(words).intersection(set(list(count_dict[l].keys())))
        if len(int_words) == len(words):
            mini = count_dict[l][words[0]]
            for word in words:
                if count_dict[l][word] < mini:
                    mini = count_dict[l][word]
            count_dict[l][ph] = mini
    return count_dict


def delete_phrase_words(count_dict, l, seed_phrases):
    if len(seed_phrases) == 0:
        return count_dict
    for ph in seed_phrases:
        words = ph.split()
        for word in words:
            try:
                del count_dict[l][word]
            except:
                pass
    return count_dict


def post_process(count_dict, l, seed_phrases):
    count_dict = add_phrase_keys(count_dict, l, seed_phrases)
    count_dict = delete_phrase_words(count_dict, l, seed_phrases)
    return count_dict


def get_train_data(df, labels, label_term_dict):
    y = []
    X = []
    y_true = []
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
        words = line.strip().split()
        count_dict = {}
        flag = 0
        for l in labels:
            seed_phrases = []
            seed_words = set()
            for w in label_term_dict[l]:
                if len(w.split()) > 1:
                    seed_phrases.append(w)
                    seed_words.update(set(w.split()))
                else:
                    seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
            count_dict = post_process(count_dict, l, seed_phrases)
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            # lbl = softmax_label(count_dict, label_to_index)
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


def get_label_term_dict(labels):
    label_term_dict = {}
    for i in labels:
        terms = i.split("_")
        if i == "stocks_and_bonds":
            label_term_dict[i] = ["stocks", "bonds"]
        elif i == "the_affordable_care_act":
            label_term_dict[i] = ["affordable", "care", "act"]
        else:
            label_term_dict[i] = terms
    return label_term_dict


def train_classifier(df, labels, label_term_dict, label_to_index, index_to_label):
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    glove_dir = basepath + "glove.6B"
    model_name = "cocube_tok"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100

    X, y, y_true = get_train_data(df, labels, label_term_dict)
    print("****************** CLASSIFICATION REPORT FOR TRAINING DATA ********************")
    print(classification_report(y_true, y))
    df_train = create_training_df(X, y, y_true)
    df_train.to_csv(basepath + dataset + "training_label.csv")
    y_one_hot = make_one_hot(y, label_to_index)
    # y = np.array(y)
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
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
    print("****************** CLASSIFICATION REPORT FOR DOCUMENTS WITH LABEL WORDS ********************")
    X_label_all = prep_data(texts=X, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                            tokenizer=tokenizer)
    pred = model.predict(X_label_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true, pred_labels))
    print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
    X_all = prep_data(texts=df["sentence"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    y_true_all = df["label"]
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true_all, pred_labels))
    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels

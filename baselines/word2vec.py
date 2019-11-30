from sklearn.metrics.pairwise import cosine_similarity
from coc_data_utils import get_label_term_json
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pickle
import numpy as np


def get_label_w2v_dict(label_term_dict):
    label_w2v_dict = {}
    for l in label_term_dict:
        temp = np.zeros((100,))
        for w in label_term_dict[l]:
            try:
                temp += model[w]
            except Exception as e:
                print("Word ", w, e)
        label_w2v_dict[l] = temp
    return label_w2v_dict


if __name__ == "__main__":
    basepath = "../data/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_tokenized_clean_child.pkl", "rb") as handler:
        df = pickle.load(handler)

    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_child_uncon.json")

    tagged_data = [word_tokenize(_d.lower()) for i, _d in enumerate(df["sentence"])]
    max_epochs = 20
    vec_size = 100
    alpha = 0.025

    # model = Word2Vec(tagged_data, size=vec_size, alpha=alpha)
    # model.save(pkl_dump_dir + "w2v.model")
    # print("Model Saved")

    model = Word2Vec.load(pkl_dump_dir + "w2v.model")

    label_w2v_dict = get_label_w2v_dict(label_term_dict)

    pred = []

    for i, row in df.iterrows():
        words = word_tokenize(row["sentence"].lower())
        temp = np.zeros((100,))
        for w in words:
            try:
                temp += model[w]
            except Exception as e:
                print("Word: ", w, e)
        maxi = -1
        max_l = ""
        for l in label_w2v_dict:
            cos = cosine_similarity((temp / len(words)).reshape(1, -1), label_w2v_dict[l].reshape(1, -1))[0][0]
            if cos > maxi:
                maxi = cos
                max_l = l
        pred.append(max_l)

    print(classification_report(df["label"], pred))

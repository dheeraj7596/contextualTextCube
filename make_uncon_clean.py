import pickle
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string


def get_child_to_parent(pkl_dump_dir, flag):
    f = open(pkl_dump_dir + "label_hier.txt")
    lines = f.readlines()
    child_to_parent = {}
    for line in lines:
        entities = line.strip().split()
        p = entities[0]
        if flag == 1:
            child_to_parent[p] = p
        for e in entities[1:]:
            child_to_parent[e] = p
    f.close()
    return child_to_parent


def modify_to_parent(df, child_to_parent):
    dic = {}
    dic["sentence"] = []
    dic["label"] = []
    for index, row in df.iterrows():
        dic["sentence"].append(row["sentence"])
        p_label = child_to_parent[row["label"]]
        dic["label"].append(p_label)
    df_X = DataFrame(dic)
    return df_X


def modify_to_child(df, child_to_parent):
    dic = {}
    dic["sentence"] = []
    dic["label"] = []
    for index, row in df.iterrows():
        l = row["label"]
        try:
            temp = child_to_parent[l]
            dic["sentence"].append(row["sentence"])
            dic["label"].append(l)
        except:
            continue

    df_X = DataFrame(dic)
    return df_X


def make_df(df):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    for index, row in df.iterrows():
        line = row["sentence"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            words = word_tokenize(sent)
            new_words = []
            for word in words:
                if word in stop_words:
                    continue
                word_clean = word.translate(str.maketrans('', '', string.punctuation))
                if len(word_clean) == 0 or word_clean in stop_words:
                    continue
                new_words.append(word_clean)
            sentences[sentence_ind] = " ".join(new_words)
        df["sentence"][index] = " ".join(sentences)
    return df


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_tokens_limit.pkl", "rb"))
    df = make_df(df)
    child_to_parent = get_child_to_parent(pkl_dump_dir, flag=1)
    df_parent = modify_to_parent(df, child_to_parent)
    child_to_parent = get_child_to_parent(pkl_dump_dir, flag=0)
    df_child = modify_to_child(df, child_to_parent)
    pass

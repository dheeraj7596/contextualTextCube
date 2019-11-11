import pickle
from nltk.corpus import stopwords
import string


def get_vec(word, word_cluster, stop_words):
    if word in stop_words:
        return None
    t = word.split("$")
    if len(t) == 1:
        prefix = t[0]
        cluster = 0
    elif len(t) == 2:
        prefix = t[0]
        cluster = t[1]
        if cluster == "1" or cluster == "0":
            cluster = int(cluster)
        else:
            prefix = word
            cluster = 0
    else:
        prefix = "".join(t[:-1])
        cluster = t[-1]
        if cluster == "1" or cluster == "0":
            cluster = int(cluster)
        else:
            cluster = 0

    word_clean = prefix.translate(str.maketrans('', '', string.punctuation))
    if len(word_clean) == 0 or word_clean in stop_words:
        return None
    try:
        vec = word_cluster[word_clean][cluster]
    except:
        try:
            vec = word_cluster[prefix][cluster]
        except:
            try:
                vec = word_cluster[word][0]
            except:
                vec = None
    return vec


def dump_clean(df, word_cluster):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    word_vec = {}
    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)))
        line = row["sentence"]
        words = line.strip().split()
        new_words = []
        for word in words:
            try:
                vec = word_vec[word]
            except:
                vec = get_vec(word, word_cluster, stop_words)
                if not vec:
                    continue
                word_vec[word] = vec
            new_words.append(word)
        df["sentence"][index] = " ".join(new_words)
    return df, word_vec


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_contextualized_clean.pkl", "rb"))
    word_cluster = pickle.load(open(pkl_dump_dir + "word_cluster_clean.pkl", "rb"))
    df, word_vec = dump_clean(df, word_cluster)

    print("Dumping df..")
    pickle.dump(df, open(pkl_dump_dir + "df_contextualized_clean_removed_stopwords.pkl", "wb"))

    df.to_excel(pkl_dump_dir + "df_contextualized_clean_removed_stopwords.xlsx")

    print("Dumping word_vec..")
    pickle.dump(word_vec, open(pkl_dump_dir + "word_vec_clean_removed_stopwords.pkl", "wb"))

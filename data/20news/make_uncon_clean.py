import pickle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string


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
        df["sentence"][index] = " . ".join(sentences)
    return df


if __name__ == "__main__":
    basepath = "./"
    pkl_dump_dir = basepath

    df = pickle.load(open(pkl_dump_dir + "df_tokens_limit.pkl", "rb"))
    df = make_df(df)

    pickle.dump(df, open("./df_tokenized_clean_child.pkl", "wb"))
    for i, row in df.iterrows():
        l = row["label"]
        df["label"][i] = l.strip().split(".")[0]

    pickle.dump(df, open("df_tokenized_limit_clean_parent.pkl", "wb"))

    pass

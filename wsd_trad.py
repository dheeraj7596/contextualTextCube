import pickle
from nltk.wsd import lesk

if __name__ == "__main__":
    basepath = "./data/"
    dataset = "nyt/"
    data_path = basepath + dataset
    df = pickle.load(open(data_path + "df_tokenized_clean_child.pkl", "rb"))
    sent = df.iloc[0]["sentence"]
    tokens = sent.strip().split()
    word_interpretation = {}
    word_counter = {}

    for i, row in df.iterrows():
        sent = row["sentence"]
        tokens = sent.strip().split()
        new_tokens = []
        for tok in tokens:
            interp = str(lesk(tokens, tok))
            if interp == None:
                new_tokens.append(tok)
                continue
            try:
                temp_word = word_interpretation[tok]
                try:
                    temp_interp = word_interpretation[tok][interp]
                except:
                    word_interpretation[tok][interp] = word_counter[tok]
                    word_counter[tok] += 1
            except:
                word_interpretation[tok] = {}
                word_counter[tok] = 1
                word_interpretation[tok][interp] = word_counter[tok]
                word_counter[tok] += 1

            new_tokens.append(tok + "$" + str(word_interpretation[tok][interp]))
        df["sentence"][i] = " ".join(new_tokens)

    pickle.dump(word_interpretation, open(data_path + "word_interpretation_lesk.pkl", "wb"))
    pickle.dump(df, open(data_path + "df_tokenized_contextualized_lesk_clean_child.pkl", "wb"))

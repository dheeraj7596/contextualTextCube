import pickle

if __name__ == "__main__":
    df = pickle.load(open("df_tokenized_contextualized_clean_parent.pkl", "rb"))

    for i, row in df.iterrows():
        l = row["label"]
        df["label"][i] = l.strip().split(".")[0]

    pickle.dump(df, open("df_tokens_limit_parent.pkl", "wb"))
    pass

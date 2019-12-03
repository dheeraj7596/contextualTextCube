import pickle

if __name__ == "__main__":
    df = pickle.load(open("./df_tokens_limit.pkl", "rb"))
    to_remove = [7632, 8184, 13555]
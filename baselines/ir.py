from coc_data_utils import get_label_term_json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import classification_report

if __name__ == "__main__":
    basepath = "../data/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_tokenized_limit_clean_parent.pkl", "rb") as handler:
        df = pickle.load(handler)

    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_parent_uncon.json")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["sentence"])
    X_arr = X.toarray()
    names = vectorizer.get_feature_names()

    label_term_index_dict = {}
    for i in label_term_dict:
        label_term_index_dict[i] = []
        for w in label_term_dict[i]:
            try:
                w = w.split("$")[0]
                label_term_index_dict[i].append(names.index(w))
            except Exception as e:
                print("Exception for: ", w, e)

    pred = []
    for i in X_arr:
        maxi = -1
        max_l = ""
        for l in label_term_index_dict:
            sum = 0
            for ind in label_term_index_dict[l]:
                sum += i[ind]
            if sum > maxi:
                maxi = sum
                max_l = l

        pred.append(max_l)

    print(classification_report(df["label"], pred))

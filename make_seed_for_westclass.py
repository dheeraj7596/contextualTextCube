from coc_data_utils import get_label_term_json
import pickle

if __name__ == "__main__":
    basepath = "./data/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset
    west_dump_dir = "/Users/dheerajmekala/Work/WeSTClass-master/" + dataset

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit_parent.pkl", "rb"))
    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_parent.json")
    f1 = open(west_dump_dir + "classes.txt", "w")
    f2 = open(west_dump_dir + "keywords.txt", "w")

    i = 0
    for l in label_term_dict:
        f1.write(str(i) + ":" + l + "\n")
        temp = []
        for p in label_term_dict[l]:
            temp.append(p.split("$")[0])
        f2.write(str(i) + ":" + ",".join(temp) + "\n")
        i += 1
    f1.close()
    f2.close()
    df.to_csv(west_dump_dir + "dataset.csv")

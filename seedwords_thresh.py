from coc_data_utils import *
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os
import pickle
import statistics


def get_thresh(word_dump_dir, seed):
    word_dir = os.path.join(word_dump_dir, seed)
    filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                 os.path.isfile(os.path.join(word_dir, o))]
    tok_vecs = []
    for path in filepaths:
        with open(path, 'rb') as fp:
            tok_vecs.append(pickle.load(fp))

    n = len(filepaths)
    pairs = list(itertools.combinations(range(n), 2))
    min_ = 5
    for p in pairs:
        sim = cosine_similarity(tok_vecs[p[0]].reshape(1, -1), tok_vecs[p[1]].reshape(1, -1))[0][0]
        min_ = min(min_, sim)
    return min_


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    word_dump_dir = pkl_dump_dir + "wordvecs_tokenized_fresh/"

    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_child.json")

    thresholds = []
    for l in label_term_dict:
        print("Label: ", l)
        for seed in label_term_dict[l]:
            print("Seed: ", seed)
            seed = seed.split("$")[0]
            t = get_thresh(word_dump_dir, seed)
            thresholds.append(t)

    print("Median is: ", statistics.median(thresholds))

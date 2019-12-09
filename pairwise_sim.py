import sys, os, pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import itertools


def pairwise_sim(word_dir):
    if not os.path.exists(word_dir):
        print("word doesn't exist ", word_dir)
        return

    print("Getting filepaths..")
    filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                 os.path.isfile(os.path.join(word_dir, o))]
    tok_vecs = []
    print("Getting vectors..", len(filepaths))
    for path in filepaths[:1500]:
        try:
            with open(path, "rb") as input_file:
                vec = pickle.load(input_file)
                tok_vecs.append(vec)
        except Exception as e:
            print("Exception: ", e)

    tuples = list(itertools.combinations(range(len(tok_vecs)), 2))
    sims = []
    for tup in tuples:
        sim = cosine_similarity(tok_vecs[tup[0]].reshape(1, -1), tok_vecs[tup[1]].reshape(1, -1))[0][0]
        sims.append(sim)
    plt.figure()
    plt.hist(sims, color='blue', edgecolor='black', bins=30)
    plt.savefig("./cluster_windows.png")


if __name__ == "__main__":
    word = sys.argv[1]
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "20news/"
    pkl_dump_dir = basepath + dataset

    word_dump_dir = pkl_dump_dir + "wordvecs_tokenized_fresh/" + word
    # word_dump_dir = pkl_dump_dir + word

    pairwise_sim(word_dump_dir)

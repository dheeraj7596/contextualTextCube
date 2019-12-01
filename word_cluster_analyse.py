import sys, os, pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def cluster(word_dir):
    if not os.path.exists(word_dir):
        print("word doesn't exist ", word_dir)
        return

    print("Getting filepaths..")
    filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                 os.path.isfile(os.path.join(word_dir, o))]
    tok_vecs = []
    print("Getting vectors..")
    for path in filepaths[:1500]:
        try:
            with open(path, "rb") as input_file:
                vec = pickle.load(input_file)
                tok_vecs.append(vec)
        except Exception as e:
            print("Exception: ", e)

    print("Fitting kmeans..")
    km = KMeans(n_clusters=2, n_jobs=-1)
    km.fit(tok_vecs)
    cc = km.cluster_centers_
    sim = cosine_similarity(cc[0].reshape(1, -1), cc[1].reshape(1, -1))[0][0]
    print("SIMILARITY: ", sim)


if __name__ == "__main__":
    word = sys.argv[1]

    basepath = "/data3/jingbo/dheeraj/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    word_dump_dir = pkl_dump_dir + "wordvecs_tokenized_new/" + word

    cluster(word_dump_dir)

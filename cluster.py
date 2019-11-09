from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import flair, torch
import os

flair.device = torch.device('cuda:3')


def cluster_all_embeddings(word_dump_dir, cluster_dump_dir, threshold=0.7):
    num_clusters = 2
    except_counter = 0
    for word_index, word in enumerate(os.listdir(word_dump_dir)):
        if word_index % 100 == 0:
            print("Finished words: " + str(word_index))
        if os.path.isdir(os.path.join(word_dump_dir, word)):
            word_dir = os.path.join(word_dump_dir, word)
            filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                         os.path.isfile(os.path.join(word_dir, o))]
            tok_vecs = []
            for path in filepaths:
                try:
                    vec = pickle.load(open(path, "rb"))
                    tok_vecs.append(vec)
                except Exception as e:
                    except_counter += 1
                    print("Exception Counter: ", except_counter, word_index, e)

            if len(tok_vecs) == 0:
                continue

            word_cluster_dump_dir = cluster_dump_dir + word
            os.makedirs(word_cluster_dump_dir, exist_ok=True)
            if len(tok_vecs) < num_clusters:
                pickle.dump(tok_vecs, open(word_cluster_dump_dir + "/cc.pkl", "wb"))
            else:
                km = KMeans(n_clusters=num_clusters, n_jobs=-1)
                km.fit(tok_vecs)
                cc = km.cluster_centers_
                sim = cosine_similarity(cc[0].reshape(1, -1), cc[1].reshape(1, -1))[0][0]
                if sim > threshold:
                    cc = [np.mean(tok_vecs, axis=0)]

                pickle.dump(cc, open(word_cluster_dump_dir + "/cc.pkl", "wb"))


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    word_dump_dir = pkl_dump_dir + "wordvecs/"
    cluster_dump_dir = pkl_dump_dir + "clusters/"

    cluster_all_embeddings(word_dump_dir, cluster_dump_dir)

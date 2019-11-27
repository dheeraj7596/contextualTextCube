from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os
import string


def cluster_all_embeddings(word_dump_dir, cluster_dump_dir, threshold=0.7):
    num_clusters = 2
    except_counter = 0
    dir_set = get_relevant_dirs(word_dump_dir)
    print("Length of DIR_SET: ", len(dir_set))

    for word_index, word in enumerate(dir_set):
        if word_index % 100 == 0:
            print("Finished words: " + str(word_index))
        if os.path.isdir(os.path.join(word_dump_dir, word)):
            word_dir = os.path.join(word_dump_dir, word)
            filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                         os.path.isfile(os.path.join(word_dir, o))]
            tok_vecs = []
            for path in filepaths[:1500]:
                try:
                    with open(path, "rb") as input_file:
                        vec = pickle.load(input_file)
                    tok_vecs.append(vec)
                except Exception as e:
                    except_counter += 1
                    print("Exception Counter: ", except_counter, word_index, e)

            if len(tok_vecs) == 0:
                print("Length of tok_vecs is 0: ", word)
                continue

            word_cluster_dump_dir = cluster_dump_dir + word
            os.makedirs(word_cluster_dump_dir, exist_ok=True)
            if len(tok_vecs) < num_clusters:
                with open(word_cluster_dump_dir + "/cc.pkl", "wb") as output_file:
                    pickle.dump(tok_vecs, output_file)
            else:
                km = KMeans(n_clusters=num_clusters, n_jobs=-1)
                km.fit(tok_vecs)
                cc = km.cluster_centers_
                sim = cosine_similarity(cc[0].reshape(1, -1), cc[1].reshape(1, -1))[0][0]
                if sim > threshold:
                    cc = [np.mean(tok_vecs, axis=0)]

                with open(word_cluster_dump_dir + "/cc.pkl", "wb") as output_file:
                    pickle.dump(cc, output_file)
        else:
            print("Not a directory: ", os.path.join(word_dump_dir, word))


def get_relevant_dirs(word_dump_dir):
    print("Getting relevant dirs..")
    dirs = os.listdir(word_dump_dir)
    dir_dict = {}
    for dir in dirs:
        dir_dict[dir] = 1

    print("Dir dict ready..")
    dir_set = set()
    for i, dir in enumerate(dirs):
        if i % 1000 == 0:
            print("Finished checking dirs: " + str(i) + " out of: " + str(len(dirs)))
        dir_new = dir.translate(str.maketrans('', '', string.punctuation))
        if len(dir_new) == 0:
            continue
        try:
            temp = dir_dict[dir_new]
            dir_set.add(dir_new)
        except:
            dir_set.add(dir)
    return dir_set


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset

    word_dump_dir = pkl_dump_dir + "wordvecs_tokenized_new/"
    cluster_dump_dir = pkl_dump_dir + "clusters_tokenized_new/"

    cluster_all_embeddings(word_dump_dir, cluster_dump_dir)

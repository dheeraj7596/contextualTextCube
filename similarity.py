import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from Word import Word

if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    word = sys.argv[1]
    print("WORD: ", word)
    pkl_dump_dir = basepath + dataset + word

    word_obj_list = pickle.load(open(pkl_dump_dir + "/word_obj_list.pkl", "rb"))
    cluster_tokvecs = {}
    cluster_tokvecs[0] = []
    cluster_tokvecs[1] = []
    for word in word_obj_list:
        if word.cluster == 0:
            cluster_tokvecs[0].append(word.tok_vec)
        else:
            cluster_tokvecs[1].append(word.tok_vec)

    sim_0 = []
    for i in range(len(cluster_tokvecs[0])):
        for ind in range(i, len(cluster_tokvecs[0])):
            sim = cosine_similarity(cluster_tokvecs[0][i].reshape(1, -1), cluster_tokvecs[0][ind].reshape(1, -1))[0][0]
            sim_0.append(sim)

    sim_1 = []
    for i in range(len(cluster_tokvecs[1])):
        for ind in range(i, len(cluster_tokvecs[1])):
            sim = cosine_similarity(cluster_tokvecs[1][i].reshape(1, -1), cluster_tokvecs[1][ind].reshape(1, -1))[0][0]
            sim_1.append(sim)

    sim_0_1 = []
    for i in range(len(cluster_tokvecs[0])):
        for ind in range(len(cluster_tokvecs[1])):
            sim = cosine_similarity(cluster_tokvecs[0][i].reshape(1, -1), cluster_tokvecs[1][ind].reshape(1, -1))[0][0]
            sim_0_1.append(sim)

    plt.figure()
    plt.hist(sim_0, color='blue', edgecolor='black', bins=30)
    plt.savefig(pkl_dump_dir + "/same_cluster_0.png")
    plt.figure()
    plt.hist(sim_1, color='blue', edgecolor='black', bins=30)
    plt.savefig(pkl_dump_dir + "/same_cluster_1.png")
    plt.figure()
    plt.hist(sim_0_1, color='blue', edgecolor='black', bins=30)
    plt.savefig(pkl_dump_dir + "/diff_cluster.png")

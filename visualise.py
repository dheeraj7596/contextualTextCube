from sklearn.manifold import TSNE
from Word import Word
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import sys
import os

sns.set_style('darkgrid')
sns.set_palette('muted')
import pickle


def fashion_scatter(x, colors, contexts, path):
    # choose a color palette with seaborn.
    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        print(ind)
        annot.xy = pos
        # i = list(x).index(ind["ind"])
        text = contexts[ind["ind"][0]]
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                f.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    f.canvas.draw_idle()

    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    marker = ["^", "o"]

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    for i in range(len(x)):
        xi = x[i][0]  # x array for ith feature .. here is where you would generalize      different x for every feature
        yi = x[i][1]  # y array for ith feature
        ci = palette[colors[i]]  # color for ith feature
        mi = marker[colors[i]]  # marker for ith feature
        plt.scatter(xi, yi, marker=mi, color=ci, edgecolors='black', s = 20*4**2)
    # sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)],
    #                 marker=marker[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.axis('off')
    plt.axis('tight')
    #
    # txts = []
    #
    # for i in range(num_classes):
    #     # Position of each label at median of data points.
    #
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     # txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     # txt.set_path_effects([
    #     #     PathEffects.Stroke(linewidth=5, foreground="w"),
    #     #     PathEffects.Normal()])
    #     # txts.append(txt)
    #
    # f.savefig(path)
    # if contexts and len(contexts) > 0:
    #     f.canvas.mpl_connect("motion_notify_event", hover)
    #     plt.show()
    plt.show()
    f.savefig(path)
    return f, ax


def visualise(tok_vecs, labels, contexts, path):
    fashion_tsne = TSNE(n_components=2).fit_transform(tok_vecs)
    f, ax = fashion_scatter(fashion_tsne, labels, contexts, path)
    return f


def get_tok_vecs(word_dir):
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
    return tok_vecs


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "20news/"
    word = "penalty"

    print("WORD: ", word)
    pkl_dump_dir = basepath + dataset + word
    word_dump_dir = pkl_dump_dir

    tok_vecs = get_tok_vecs(word_dump_dir)
    # km = pickle.load(open("./data/20news/km.pkl", "rb"))

    km = KMeans(n_clusters=2, n_jobs=-1)
    km.fit(tok_vecs)

    # tok_vecs = []
    # contexts = []
    # for word in word_obj_list:
    #     tok_vecs.append(word.tok_vec)
    #     # contexts.append(word.context)
    contexts = []
    f = visualise(tok_vecs, km.labels_, contexts, pkl_dump_dir + "/plot.png")

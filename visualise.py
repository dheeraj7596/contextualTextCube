from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
import pickle


class Word():
    def __init__(self, name, context, tok_vec, label, cluster=None):
        self.name = name
        self.context = context
        self.tok_vec = tok_vec
        self.label = label
        self.cluster = cluster


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def visualise(tok_vecs, labels):
    fashion_tsne = TSNE(n_components=2).fit_transform(tok_vecs)
    f, ax, sc, txts = fashion_scatter(fashion_tsne, labels)
    return f


if __name__ == "__main__":
    basepath = "./data/"
    dataset = "nyt/"
    word = "penalty"

    print("WORD: ", word)
    pkl_dump_dir = basepath + dataset + word

    word_obj_list = pickle.load(open(pkl_dump_dir + "/word_obj_list.pkl", "rb"))
    km = pickle.load(open(pkl_dump_dir + "/km.pkl", "rb"))
    tok_vecs = []
    for word in word_obj_list:
        tok_vecs.append(word.tok_vec)

    f = visualise(tok_vecs, km.labels_)

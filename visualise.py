from sklearn.manifold import TSNE
from Word import Word
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import sys

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

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
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

    f.savefig(path)
    if contexts and len(contexts) > 0:
        f.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
    return f, ax, sc, txts


def visualise(tok_vecs, labels, contexts, path):
    fashion_tsne = TSNE(n_components=2).fit_transform(tok_vecs)
    f, ax, sc, txts = fashion_scatter(fashion_tsne, labels, contexts, path)
    return f


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    word = sys.argv[1]

    print("WORD: ", word)
    pkl_dump_dir = basepath + dataset + word

    word_obj_list = pickle.load(open(pkl_dump_dir + "/word_obj_list.pkl", "rb"))
    km = pickle.load(open(pkl_dump_dir + "/km.pkl", "rb"))
    tok_vecs = []
    contexts = []
    for word in word_obj_list:
        tok_vecs.append(word.tok_vec)
        contexts.append(word.context)

    f = visualise(tok_vecs, km.labels_, contexts, pkl_dump_dir + "/plot.png")

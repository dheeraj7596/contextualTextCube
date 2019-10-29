import pickle
from flair.data import Sentence


def get_word_context(token_ind, sentence):
    sent = Sentence(sentence)
    word = sent.tokens[token_ind]
    if token_ind - 2 >= 0:
        left = token_ind - 2
    else:
        left = 0
    if token_ind + 2 < len(sent.tokens):
        right = token_ind + 2
    else:
        right = len(sent.tokens)
    context = ""
    for token in sent.tokens[left:right]:
        context = context + " " + token.text
    return word.text.strip().lower(), context


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    print("Loaded pickle dumps..")
    sentences = pickle.load(open(pkl_dump_dir + "sentences.pkl", "rb"))
    km = pickle.load(open(pkl_dump_dir + "km.pkl", "rb"))
    mapping = pickle.load(open(pkl_dump_dir + "mapping.pkl", "rb"))
    labels = km.labels_

    print("Creating word_to_cluster_context map..")
    word_to_cluster_context = {}
    for i, label in enumerate(labels):
        if i % 10000 == 0:
            print("Finished token: " + str(i) + " out of " + str(len(labels)))
        token_ind = mapping[i]["token"]
        sentence_ind = mapping[i]["sentence"]
        word, context = get_word_context(token_ind, sentences[sentence_ind])
        if word in word_to_cluster_context:
            if label in word_to_cluster_context[word]:
                word_to_cluster_context[word][label].append(context)
            else:
                word_to_cluster_context[word][label] = [context]
        else:
            word_to_cluster_context[word] = {}
            word_to_cluster_context[word][label] = [context]

    print("Creating several_cluster_words map..")
    several_cluster_words = {}
    for word in word_to_cluster_context:
        if len(word_to_cluster_context[word].keys()) > 1:
            several_cluster_words[word] = word_to_cluster_context[word]

    print("Dumping the maps..")
    pickle.dump(word_to_cluster_context, open(pkl_dump_dir + "word_to_cluster_context.pkl", "wb"))
    pickle.dump(several_cluster_words, open(pkl_dump_dir + "several_cluster_words.pkl", "wb"))

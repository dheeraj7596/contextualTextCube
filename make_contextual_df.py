from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pickle
import flair, torch

flair.device = torch.device('cuda:1')


def get_cluster(tok_vec, cc):
    sim0 = cosine_similarity(cc[0].reshape(1, -1), tok_vec.reshape(1, -1))[0][0]
    sim1 = cosine_similarity(cc[1].reshape(1, -1), tok_vec.reshape(1, -1))[0][0]
    if sim0 > sim1:
        return 0
    else:
        return 1


def to_tokenized_string(sentence):
    tokenized = " ".join([t.text for t in sentence.tokens])
    return tokenized


def make_word_cluster(df, embedding, cluster_dump_dir):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    except_counter = 0
    word_cluster = {}

    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)))
        line = row["sentence"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            sentence = Sentence(sent)
            embedding.embed(sentence)
            for token_ind, token in enumerate(sentence):
                word = token.text
                if word in stop_words:
                    continue
                word_cluster_dump_dir = cluster_dump_dir + word
                if word not in word_cluster:
                    try:
                        cc = pickle.load(open(word_cluster_dump_dir + "/cc.pkl", "rb"))
                        word_cluster[word] = cc
                    except Exception as e:
                        except_counter += 1
                        print("Exception Counter: ", except_counter, index, e)
                        continue
                else:
                    cc = word_cluster[word]
                if len(cc) == 2:
                    tok_vec = token.embedding.cpu().numpy()
                    cluster = get_cluster(tok_vec, cc)
                    sentence.tokens[token_ind].text = word + "$" + str(cluster)
            sentences[sentence_ind] = to_tokenized_string(sentence)
        df["sentence"][index] = " ".join(sentences)
    return df, word_cluster


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    cluster_dump_dir = pkl_dump_dir + "clusters/"

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))
    df, word_cluster = make_word_cluster(df, embedding, cluster_dump_dir)

    print("Dumping df..")
    pickle.dump(df, open(pkl_dump_dir + "df_contextualized.pkl", "wb"))

    print("Dumping word_cluster..")
    pickle.dump(word_cluster, open(pkl_dump_dir + "word_cluster.pkl", "wb"))

from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pickle
import flair, torch
import string

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
            sentence = Sentence(sent, use_tokenizer=True)
            embedding.embed(sentence)
            for token_ind, token in enumerate(sentence):
                word = token.text
                if word in stop_words:
                    continue
                word_clean = word.translate(str.maketrans('', '', string.punctuation))
                if len(word_clean) == 0 or word_clean in stop_words:
                    continue

                try:
                    cc = word_cluster[word_clean]
                except:
                    try:
                        cc = word_cluster[word]
                    except:
                        word_clean_path = cluster_dump_dir + word_clean + "/cc.pkl"
                        word_path = cluster_dump_dir + word + "/cc.pkl"
                        try:
                            with open(word_clean_path, "rb") as handler:
                                cc = pickle.load(handler)
                            word_cluster[word_clean] = cc
                        except:
                            try:
                                with open(word_path, "rb") as handler:
                                    cc = pickle.load(handler)
                                word_cluster[word] = cc
                            except Exception as e:
                                except_counter += 1
                                print("Exception Counter: ", except_counter, index, e)
                                continue

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
    dataset = "arxiv/"
    pkl_dump_dir = basepath + dataset
    cluster_dump_dir = pkl_dump_dir + "clusters_tokenized_new/"

    with open(pkl_dump_dir + "/df_tokens_limit_new.pkl", "rb") as handler:
        df = pickle.load(handler)
    df, word_cluster = make_word_cluster(df, embedding, cluster_dump_dir)

    print("Dumping df..")
    with open(pkl_dump_dir + "df_tokenized_contextualized_clean.pkl", "wb") as handler:
        pickle.dump(df, handler)

    df.to_excel(pkl_dump_dir + "df_tokenized_contextualized_clean.xlsx")

    print("Dumping word_cluster..")
    with open(pkl_dump_dir + "word_cluster_tokenized_clean.pkl", "wb") as handler:
        pickle.dump(word_cluster, handler)

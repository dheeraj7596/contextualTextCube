from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from nltk import sent_tokenize
from nltk.corpus import stopwords
import pickle
import flair, torch
import os

flair.device = torch.device('cuda:1')


def get_all_embeddings(df, embedding, pkl_dump_dir):
    stop_words = set(stopwords.words('english'))
    stop_words.add("would")
    except_counter = 0

    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished sentences: " + str(index) + " out of " + str(len(df)))
        line = row["sentence"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            sentence = Sentence(sent)
            try:
                embedding.embed(sentence)
            except Exception as e:
                except_counter += 1
                print("Exception Counter: ", except_counter, sentence_ind, index, e)
                continue
            for token_ind, token in enumerate(sentence):
                word = token.text
                if word in stop_words:
                    continue
                fname = pkl_dump_dir + word + "_.pkl"
                vec = token.embedding.cpu().numpy()
                if os.path.isfile(fname):
                    bert_vecs = pickle.load(open(fname, "rb"))
                else:
                    bert_vecs = []
                if len(bert_vecs) > 10000:
                    continue
                bert_vecs.append(vec)
                pickle.dump(bert_vecs, open(fname, "wb"))


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    word_dump_dir = basepath + dataset + "wordvecs/"

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))
    get_all_embeddings(df, embedding, word_dump_dir)

from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from nltk import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import pickle
import flair, torch
import os

flair.device = torch.device('cuda:3')


def get_all_embeddings(df, embedding, pkl_dump_dir):
    word_counter = defaultdict(int)
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
                if word in stop_words or "/" in word:
                    continue
                dump_dir = pkl_dump_dir + word
                os.makedirs(dump_dir, exist_ok=True)
                fname = dump_dir + "/" + str(word_counter[word]) + ".pkl"
                word_counter[word] += 1
                vec = token.embedding.cpu().numpy()
                try:
                    pickle.dump(vec, open(fname, "wb"))
                except Exception as e:
                    except_counter += 1
                    print("Exception Counter: ", except_counter, sentence_ind, index, word, e)


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    word_dump_dir = basepath + dataset + "wordvecs/"

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))
    get_all_embeddings(df, embedding, word_dump_dir)

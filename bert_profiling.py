from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from nltk import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import pickle
import flair, torch
import string
import os
import time

flair.device = torch.device('cuda:1')


def get_all_embeddings(df, embedding, pkl_dump_dir):
    word_counter = defaultdict(int)
    stop_words = set(stopwords.words('english'))
    stop_words.add("would")
    except_counter = 0

    for index, row in df.iterrows():
        start_ = time.time()
        if index % 100 == 0:
            print("Finished sentences: " + str(index) + " out of " + str(len(df)))
        line = row["sentence"]
        start = time.time()
        sentences = sent_tokenize(line)
        print("Sentence tokenization time: ", time.time() - start)
        for sentence_ind, sent in enumerate(sentences):
            start = time.time()
            sentence = Sentence(sent, use_tokenizer=True)
            print("Sentence time: ", time.time() - start)
            try:
                start = time.time()
                embedding.embed(sentence)
                print("Embedding time: ", time.time() - start)
            except Exception as e:
                except_counter += 1
                print("Exception Counter: ", except_counter, sentence_ind, index, e)
                continue
            for token_ind, token in enumerate(sentence):
                word = token.text
                start = time.time()
                word = word.translate(str.maketrans('', '', string.punctuation))
                print("Translation time: ", time.time() - start)
                start = time.time()
                if word in stop_words or "/" in word or len(word) == 0 or word_counter[word] >= 1500:
                    continue
                print("IF check time", time.time() - start)
                dump_dir = pkl_dump_dir + word
                # os.makedirs(dump_dir, exist_ok=True)
                # fname = dump_dir + "/" + str(word_counter[word]) + ".pkl"
                word_counter[word] += 1
                start = time.time()
                vec = token.embedding.cpu().numpy()
                print("Vec: ", time.time() - start)
                # try:
                #     pickle.dump(vec, open(fname, "wb"))
                # except Exception as e:
                #     except_counter += 1
                #     print("Exception Counter: ", except_counter, sentence_ind, index, word, e)
        print("Total sentence time: ", time.time() - start_)


if __name__ == "__main__":
    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    word_dump_dir = basepath + dataset + "wordvecs_tokenized_fresh/"

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))
    get_all_embeddings(df, embedding, word_dump_dir)

from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from Word import Word
from analyse import get_word_context
from nltk import sent_tokenize
import pickle
import flair, torch

flair.device = torch.device('cuda:1')


def get_all_embeddings(df, embedding):
    except_counter = 0
    word_obj_list = []
    word_sent_token_dict = {}

    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished sentences: " + str(index) + " out of " + str(len(df)))
        word_sent_token_dict[index] = {}
        line = row["sentence"]
        label = row["label"]
        sentences = sent_tokenize(line)
        for sentence_ind, sent in enumerate(sentences):
            word_sent_token_dict[index][sentence_ind] = {}
            sentence = Sentence(sent)
            try:
                embedding.embed(sentence)
            except Exception as e:
                except_counter += 1
                print("Exception Counter: ", except_counter, sentence_ind, index, e)
                continue
            for token_ind, token in enumerate(sentence):
                vec = token.embedding.cpu().numpy()
                word_obj = Word(name=token.text,
                                context=get_word_context(token_ind, sent),
                                tok_vec=vec,
                                label=label
                                )
                word_sent_token_dict[index][sentence_ind][token_ind] = word_obj
                word_obj_list.append(word_obj)
    return word_obj_list, word_sent_token_dict


if __name__ == "__main__":
    import sys

    embedding = BertEmbeddings('bert-base-uncased')
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset
    start = str(sys.argv[1])
    end = str(sys.argv[2])

    df = pickle.load(open(pkl_dump_dir + "/df_tokens_limit.pkl", "rb"))
    word_obj_list, word_sent_token_dict = get_all_embeddings(df[int(start): int(end)], embedding)
    pickle.dump(word_obj_list, open(pkl_dump_dir + "wordvecs/bert_vecs_" + str(start) + "_" + str(end) + ".pkl", "wb"))
    pickle.dump(word_sent_token_dict,
                open(pkl_dump_dir + "wordvecs/word_sent_token_dict_" + str(start) + "_" + str(end) + ".pkl", "wb"))

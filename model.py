import os
from data_utils import *
from keras.preprocessing.text import Tokenizer


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    return tokenizer


def create_embedding_matrix(glove_dir, tokenizer, embedding_dim):
    embeddings = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings[word] = coefs

    # Initialize a matrix to hold the word embeddings
    embedding_matrix = np.random.random(
        (len(tokenizer.word_index) + 1, embedding_dim)
    )

    # Let the padded indices map to zero-vectors. This will
    # prevent the padding from influencing the results
    embedding_matrix[0] = 0

    # Loop though all the words in the word_index and where possible
    # replace the random initalization with the GloVe vector.
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


if __name__ == "__main__":
    dataset = "nyt"
    glove_dir = "/Users/dheerajmekala/Work/Hier/Asym/glove.6B"
    embedding_dim = 100
    max_words = 20000
    pass

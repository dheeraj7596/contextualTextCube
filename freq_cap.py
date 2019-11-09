import os
import sys


def analyse(word_dump_dir, freq):
    count = 0
    for word_index, word in enumerate(os.listdir(word_dump_dir)):
        if os.path.isdir(os.path.join(word_dump_dir, word)):
            word_dir = os.path.join(word_dump_dir, word)
            filepaths = [os.path.join(word_dir, o) for o in os.listdir(word_dir) if
                         os.path.isfile(os.path.join(word_dir, o))]
            if len(filepaths) >= freq:
                count += 1
    return count


if __name__ == "__main__":
    basepath = "/data3/jingbo/dheeraj/"
    dataset = "nyt/"

    freq = int(sys.argv[1])
    pkl_dump_dir = basepath + dataset

    word_dump_dir = pkl_dump_dir + "wordvecs/"

    num_docs = analyse(word_dump_dir, freq)
    print(num_docs)

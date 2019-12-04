from sklearn.datasets import fetch_20newsgroups
import json

if __name__ == "__main__":
    dic = {}
    train = fetch_20newsgroups(subset='train')
    labels = list(train.target_names)

    for l in labels:
        dic[l] = [l]

    with open('./seedwords.json', 'w') as fp:
        json.dump(dic, fp)

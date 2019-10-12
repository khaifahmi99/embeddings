import numpy as np



from ge.classify import read_node_label,Classifier

from ge import Node2Vec

from sklearn.linear_model import LogisticRegression

import pandas as pd

import pickle



import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE



def evaluate_embeddings(embeddings):

    X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)

    tr_frac = 0.8

    print("Training classifier using {:.2f}% nodes...".format(

        tr_frac * 100))

    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    clf.split_train_evaluate(X, Y, tr_frac)





def plot_embeddings(embeddings,):

    X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)



    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)



    color_idx = {}

    for i in range(len(X)):

        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)



    for c, idx in color_idx.items():

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()

    plt.show()

if __name__ == "__main__":

    # df = pd.read_csv('dataframe.csv')

    # weapon_type = df['weapon'].unique()
    # factors = weapon_type

    G = nx.read_edgelist('input/all.edgelist', delimiter="-----", create_using=nx.DiGraph())

    model = Node2Vec(G, 10, 80, workers=1,p=0.25,q=2 )
    model.train()
    embeddings = model.get_embeddings()

    print(embeddings)

    # report_key = []

    # for emb in embeddings:
    #     if emb not in factors:
    #         report_key.append(emb)

    # embs = [embeddings[x] for x in report_key]

    # key = np.array(report_key)
    # val = np.array(embs)

    # fin = dict(zip(key, val))
    def save_obj(obj, name):
        with open('embeddings/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    save_obj(embeddings, 'all')
    # np.save('keys/node2vec', key)

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
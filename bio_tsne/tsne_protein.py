import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

import pickle
import os

class BioTsne:
    def __init__(self):
        print 'TSNE is running..'

    # making tsne
    def make_tsne(self, model):
        if not os.path.isfile("./bio_tsne/tsne.p"):
            # make tsne
            X = model[model.wv.vocab]
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)

            # save X_tsne
            f = open("./bio_tsne/tsne.p","wb")
            pickle.dump(X_tsne , f)

            f.close()

    def visualization(self):
        # load X_tsne data
        f = open( "./bio_tsne/tsne.p" , "rb")
        X_tsne = pickle.load(f)

        plt.scatter(X_tsne[:,0], X_tsne[:, 1])
        plt.show()

        f.close()

    def link_labels_2Dim(self, model.wv.vocab, ):
        #link labels with 2-dimension vectors
        dictionary = dict()
        labels = model.wv.vocab.keys()
        f = open( "./bio_tsne/tsne.p", "rb")
        X_tsne = pickle.load(f)
        for i, label in enumerate(labels):
            x, y = x_tsne[i, :]
            color_value = choose_category(calculate_property(label), "mass")
            plt.scatter(x, y, c = color_value)


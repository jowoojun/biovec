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
        if not os.path.isfile("./trained_models/protein_2D_vector"):
            # make tsne # have to use csv file
            # X = model[model.wv.vocab]
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)

            # save X_tsne
            f = open("./trained_models/protein_2D_vector","wb")
            pickle.dump(X_tsne , f)

            f.close()

    def visualization(self):
        # load X_tsne data
        f = open( "./trained_models/protein_2D_vector" , "rb")
        X_tsne = pickle.load(f)

        plt.scatter(X_tsne[:,0], X_tsne[:, 1])
        plt.show()

        f.close()


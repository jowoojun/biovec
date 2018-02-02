import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pickle
import os

class BioTsne:
    def __init__(self):
        print 'TSNE is running..'

    # making tsne
    def make_tsne(self, model):
        if not os.path.isfile("./trained_models/ngram_2D_vector"):
            # make tsne
            X = model[model.wv.vocab]
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)

            # save X_tsne
            f = open("./trained_models/ngram_2D_vector","wb")
            pickle.dump(X_tsne , f)

            f.close()

    def visualization(self):
        # load X_tsne data
        f = open( "./trained_models/ngram_2D_vector" , "rb")
        X_tsne = pickle.load(f)
        
        #set marker size
        marker_size=10
        
        #set scatter 
        plt.scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        
        #set color bar 
        #cbar=plt.colorbar()
        
        #set label
        plt.title("Mass")
            
        plt.show()

        f.close()

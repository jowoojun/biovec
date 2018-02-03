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

    def link_with_vector(self, vectors, property_dict):
        for i, label in enumerate(property_dict.keys()):
            property_dict[label] = vectors[i] + [property_dict[label]]
        return property_dict

    def visualization(self):
        # load X_tsne data
        f = open( "./trained_models/ngram_2D_vector" , "rb")
        X_tsne = pickle.load(f)
        


        fig ,axarr = plt.subplots(2,3)
        
        #set marker size
        marker_size=1

        #set scatter 
        axarr[0,0].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[0,0].set_title("Mass")

        axarr[0,1].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[0,1].set_title("Volume")

        axarr[0,2].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[0,2].set_title("Van der Waals Volume")
        
        axarr[1,0].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[1,0].set_title("Polarity")
        
        axarr[1,1].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[1,1].set_title("Hydrophobicity")
        
        axarr[1,2].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size )#,"""X_tsne[:,2]""")
        axarr[1,2].set_title("Charge")
        
        #set color bar 
        #cbar=plt.colorbar()
        
        plt.show()

        f.close()

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
        file_path="./trained_models/ngram_2D_vector"
        if not os.path.isfile(file_path):
            # make tsne
            X = model[model.wv.vocab]
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)

            # save X_tsne
            f = open(file_path,"wb")
            pickle.dump(X_tsne , f)

            f.close()

    def density_tsne(self , file_path , model):
        if not os.path.isfile(file_path):
            print model
            # make tsne
            X = model[model.wv.vocab]
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)
            
            # save X_tsne
            f = open(file_path,"wb")
            pickle.dump(X_tsne , f)
            
            f.close()

    def link_with_vector(self, vectors, property_list):
        print np.append(vectors , property_list,axis=1)
        return np.append(vectors , property_list,axis=1)

    def visualization(self, X_tsne):
        # load final_embedding data
        
        fig , axarr = plt.subplots(2,3, figsize=(15,10))
        #set marker size
        marker_size=1

        #set scatter
        g1 = axarr[0,0].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,2])
        axarr[0,0].set_title("Mass")
        fig.colorbar(g1, ax=axarr[0,0])

        g2 = axarr[0,1].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,3])
        axarr[0,1].set_title("Volume")
        fig.colorbar(g2, ax=axarr[0,1])

        g3 = axarr[0,2].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,4])
        axarr[0,2].set_title("Van der Waals Volume")
        fig.colorbar(g3, ax=axarr[0,2])

        g4 = axarr[1,0].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,5])
        axarr[1,0].set_title("Polarity")
        fig.colorbar(g4, ax=axarr[1,0])

        g5 = axarr[1,1].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,6])
        axarr[1,1].set_title("Hydrophobicity")
        fig.colorbar(g5, ax=axarr[1,1])

        g6 = axarr[1,2].scatter(X_tsne[:,0], X_tsne[:, 1], marker_size ,X_tsne[:,7])
        axarr[1,2].set_title("Charge")
        fig.colorbar(g6, ax=axarr[1,2])

        plt.show()

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
    def make_tsne(self):
        if not os.path.isfile("./trained_models/protein_2D_vector"):
            # make tsne # have to use csv file
            #protein csv parsing
            print "Loading protvec"
            path='./trained_models/protein_vector.csv'

            vectors_float = []
            with open(path) as protein_vector:
                for line in protein_vector:
                    uniprot_id, vector = line.rstrip().split('\t', 1)
                    vectors_float.append(map(float, vector.split()))

            vectors_array = np.array(vectors_float,ndmin=2)
            vectors_float = None

            print vectors_array
            print "... OK\n"

            vectors_array=np.nan_to_num(vectors_array)
            
            print "Making tsne"
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(vectors_array)
            print "... OK\n"

            print "Saving tsne"
            # save X_tsne
            f = open("./trained_models/protein_2D_vector","wb")
            pickle.dump(X_tsne , f)
            f.close()
            print "... OK\n"

    def visualization(self):
        # load X_tsne data
        f = open( "./trained_models/protein_2D_vector" , "rb")
        X_tsne = pickle.load(f)

        plt.scatter(X_tsne[:,0], X_tsne[:, 1])
        plt.show()

        f.close()
        return 0


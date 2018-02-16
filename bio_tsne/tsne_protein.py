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
            file_path='./trained_models/protein_vector.csv'
            families = []
            vectors = []
            with open(file_path) as infile:
                for line in infile:
                    uniprot_id ,  vector_string = line.rstrip().split('\t', 2)
                    if 
                    vectors.append(np.array(map(float, vector_string.split()), dtype=np.float32))  

            vectors_array = np.array(vectors)
            vectors = None
            print "... OK\n"

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


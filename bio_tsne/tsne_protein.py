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


    # create array
    def csv_to_array(self,path):
        #protein csv parsing
        print "Loading protvec"

        vectors_float = []
        with open(path) as protein_vector:
            for line in protein_vector:
                uniprot_id, vector = line.rstrip().split('\t', 1)
                vectors_float.append(map(float, vector.split()))

        vectors_array = np.array(vectors_float,ndmin=2,dtype=np.float32)
        vectors_float = None
        vectors_array=np.nan_to_num(vectors_array)
        
        print vectors_array
        print "... OK\n"

        return vectors_array
        
    # making tsne
    def make_tsne(self,file_path,vectors_array):
        # make tsne 
        print "Making tsne"
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(vectors_array)
        print "... OK\n"

        print "Saving tsne"
        # save X_tsne
        
        f = open(file_path,"wb")
        pickle.dump(X_tsne , f)
        f.close()
        print "... OK\n"

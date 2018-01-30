import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

import pickle

class BioTsne:
	def __init__(self):
		print 'TSNE is running..'

        # making tsne
	def make_tsne(self, model):
		X = model[model.wv.vocab]

		tsne = TSNE(n_components=2)
		X_tsne = tsne.fit_transform(X)

                # save X_tsne
                pickle.dump(X_tsne , open("tsne.p","wb") )


        def visualization(self):
                # load X_tsne data
                X_tsne = pickle.load( open( "tsne.p" , "rb") )
		plt.scatter(X_tsne[:,0], X_tsne[:, 1])
		plt.show()



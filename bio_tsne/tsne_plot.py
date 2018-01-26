import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

class Tsne:
	def visualization(model):
		X = model[model.vocab]

		tsne = TSNE(n_components=2)
		X_tsne = tsne.fit_transform(X)

		plt,scatter(X_tsne[:,0], X_tsne[:, 1])
		plt.show()

		

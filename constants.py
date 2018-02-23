from gensim.models import word2vec
from scipy import linalg
import numpy as np
models = word2vec.Word2Vec.load("./trained_model/ngram_model")
vectors = models[models.wv.vocab]
def normalize(vector):
    return vector / np.sqrt(np.dot(vector, vector))
norm_vectors = map(normalize, vectors)
#norm_vectors = vectors / np.sqrt(np.dot(vectors, vectors))
print linalg.svdvals(norm_vectors)[0]


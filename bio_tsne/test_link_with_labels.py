import numpy as np
from ngrams_properties import *
from bio_tsne import tsne_3gram
from tsne_3gram import BioTsne
import collections

def test_link_with_labels():
    vectors = np.array([[0.5, 0.1], [0.2, 0.3], [0.5, 0.3], [0.8, 0.2]])
    labels_with_properties = []
    labels_with_properties += [[1, 2, 3, 4, 5, 6]]
    labels_with_properties += [[2, 3, 4, 5, 6, 7]]
    labels_with_properties += [[3, 4, 5, 6, 7, 8]]
    labels_with_properties += [[4, 5, 6, 7, 8, 9]]
    bio = BioTsne()
    result = bio.link_with_vector(vectors, labels_with_properties)
    print result.shape
    array = np.array([ [0.5, 0.1, 1., 2., 3., 4., 5., 6.],
                       [0.2, 0.3, 2., 3., 4., 5., 6., 7.],
                       [0.5, 0.3, 3., 4., 5., 6., 7., 8.],
                       [0.8, 0.2, 4., 5., 6., 7., 8., 9.]])
    print array.shape
    assert result == array

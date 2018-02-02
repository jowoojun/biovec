import numpy as np
from ngrams_properties import *
from bio_tsne import tsne_3gram
from tsne_3gram import BioTsne

def test_link_with_labels():
    vectors = [[0.5, 0.1], [0.2, 0.3]]
    labels_with_properties = {'ACD' : 1, 'CDI' : 2}
    bio = BioTsne()
    result = bio.link_with_vector(vectors, labels_with_properties)
    assert result['ACD'] == [0.5, 0.1, 1]
    assert result['CDI'] == [0.2, 0.3, 2]

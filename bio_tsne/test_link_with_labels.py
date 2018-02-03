import numpy as np
from ngrams_properties import *
from bio_tsne import tsne_3gram
from tsne_3gram import BioTsne
import collections

def test_link_with_labels():
    vectors = [[0.5, 0.1], [0.2, 0.3], [0.5, 0.3], [0.8, 0.2]]
    labels_with_properties = collections.OrderedDict()
    labels_with_properties['ACD'] = 1
    labels_with_properties['CDI'] = 2
    labels_with_properties['CCC'] = 3
    labels_with_properties['DDD'] = 4
    bio = BioTsne()
    result = bio.link_with_vector(vectors, labels_with_properties)
    assert result['ACD'] == [0.5, 0.1, 1]
    assert result['CDI'] == [0.2, 0.3, 2]
    assert result['CCC'] == [0.5, 0.3, 3]
    assert result['DDD'] == [0.8, 0.2, 4]

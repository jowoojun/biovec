from ngrams_properties import calculate_property
from ngrams_properties import make_property_dict
from ngrams_properties import choose_category
from ngrams_properties import NGRAM_PROPERTIES
import numpy as np

def test_NGRAM_PROPERTIES():
    assert [1, 2, 3, 4, 5, 6] == NGRAM_PROPERTIES['A']
    assert [1, 2, 3, 4, 5, 6] == NGRAM_PROPERTIES['C']
def test_calculate_property():
    label = 'ACD'
#    assert np.array([3, 6, 9, 12, 15, 18]) == calculate_property(label)
    assert np.array_equal(np.array([3, 6, 9, 12, 15, 18]),
                          calculate_property(label))
def test_make_property_dict():
    labels = ['ACD', 'DEF', 'HIK']
#    assert np.array([3, 6, 9, 12, 15, 18]) == make_property_dict(labels)['ACD']
#    assert np.array([3, 6, 9, 12, 15, 18]) == make_property_dict(labels)['DEF']
#    assert np.array([3, 6, 9, 12, 15, 18]) == make_property_dict(labels)['HIK']
    assert np.array_equal(np.array([3, 6, 9, 12, 15, 18]), make_property_dict(labels)['ACD'])
    assert np.array_equal(np.array([3, 6, 9, 12, 15, 18]), make_property_dict(labels)['DEF'])
    assert np.array_equal(np.array([3, 6, 9, 12, 15, 18]), make_property_dict(labels)['HIK'])
def test_choose_category():
    sum_properties = np.array([3, 6, 9, 12, 15, 18])
    category_m = "mass"
    category_v = "volume"
    assert 3 == choose_category(sum_properties, category_m)
    assert 6 == choose_category(sum_properties, category_v)


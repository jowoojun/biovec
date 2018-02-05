from ngrams_properties import calculate_property
from ngrams_properties import make_property_list
from ngrams_properties import NGRAM_PROPERTIES
from ngrams_properties import pick_key
import numpy as np

#  def test_NGRAM_PROPERTIES():
#      assert [1, 2, 3, 4, 5, 6] == NGRAM_PROPERTIES['A']
#      assert [1, 2, 3, 4, 5, 6] == NGRAM_PROPERTIES['C']
#
#  def test_calculate_property():
#      label = 'ACD'
#  #    assert np.array([3, 6, 9, 12, 15, 18]) == calculate_property(label)
#      assert np.array_equal(np.array([3, 6, 9, 12, 15, 18]),
#                            calculate_property(label))
#  def test_make_property_dict():
#      labels = ['ACD', 'DEF', 'HIK']
#      assert np.array_equal([3, 6, 9, 12, 15, 18], make_property_list(labels)[0])
#      assert np.array_equal([3, 6, 9, 12, 15, 18], make_property_list(labels)[1])
#      assert np.array_equal([3, 6, 9, 12, 15, 18], make_property_list(labels)[2])


def test_pick_key():
    assert pick_key('B') == NGRAM_PROPERTIES['N'] or pick_key('B') == NGRAM_PROPERTIES['D']

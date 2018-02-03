import numpy as np
import collections

NGRAM_PROPERTIES = {
    'A': [1, 2, 3, 4, 5, 6],
    'C': [1, 2, 3, 4, 5, 6],
    'D': [1, 2, 3, 4, 5, 6],
    'E': [1, 2, 3, 4, 5, 6],
    'F': [1, 2, 3, 4, 5, 6],
    'G': [1, 2, 3, 4, 5, 6],
    'H': [1, 2, 3, 4, 5, 6],
    'I': [1, 2, 3, 4, 5, 6],
    'K': [1, 2, 3, 4, 5, 6],
    'L': [1, 2, 3, 4, 5, 6],
    'M': [1, 2, 3, 4, 5, 6],
    'N': [1, 2, 3, 4, 5, 6],
    'P': [1, 2, 3, 4, 5, 6],
    'Q': [1, 2, 3, 4, 5, 6],
    'R': [1, 2, 3, 4, 5, 6],
    'S': [1, 2, 3, 4, 5, 6],
    'T': [1, 2, 3, 4, 5, 6],
    'V': [1, 2, 3, 4, 5, 6],
    'W': [1, 2, 3, 4, 5, 6],
    'Y': [1, 2, 3, 4, 5, 6],
    'X': [1, 2, 3, 4, 5, 6]
}

# sum up 3grams property
"""
sum_properties = property['A'] + property['B'] + property['C']
"""
def calculate_property(label):
    split_to_char = list(label)
    sum_properties = np.array([0, 0, 0, 0, 0, 0])
    for char in split_to_char:
        sum_properties += np.array(NGRAM_PROPERTIES[char])
    return sum_properties


"""
list = [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7]
                          ]
"""
def make_property_list(labels):
    property_list = []
    for label in labels:
        property_list += [calculate_property(label)]
    return property_list

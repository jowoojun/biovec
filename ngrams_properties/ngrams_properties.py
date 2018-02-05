import numpy as np
import collections
from random import *

NGRAM_PROPERTIES = {
    'A': [71.0788  , 67. , 67. , 7.   , 47. , 6.01 ],
    'C': [103.1448 , 86. , 86. , 8.   , 52. , 5.05 ],
    'D': [115.0886 , 91. , 91. , 18.  , -18., 2.85 ],
    'E': [129.1155 , 109., 109., 17.  , -31., 3.15 ],
    'F': [147.1766 , 135., 135., 4.   , 100., 5.49 ],
    'G': [57.052   , 48. , 48. , 9.   , 0.  , 6.06 ],
    'H': [137.1412 , 118., 118., 13.  , -42., 7.6  ],
    'I': [113.1595 , 124., 124., 2.   , 99. , 6.05 ],
    'K': [128.1742 , 135., 135., 15.  , -23., 9.6  ],
    'L': [113.1595 , 124., 124., 1.   , 100., 6.01 ],
    'M': [131.1986 , 124., 124., 5.   , 74. , 5.74 ],
    'N': [114.1039 , 96. , 96. , 16.  , -41., 5.41 ],
    'P': [97.1167  , 90. , 90. , 11.5 , -46., 6.3  ],
    'Q': [128.1308 , 114., 114., 14.  , 8.  , 5.65 ],
    'R': [156.1876 , 148., 148., 19.  , 41. , 10.76],
    'S': [87.0782  , 73. , 73. , 12.  ,-7.  , 5.68 ],
    'T': [101.1051 , 93. , 93. , 11.  , 13. , 5.6  ],
    'V': [99.1326  , 105., 105., 3.   , 79. , 6.   ],
    'W': [186.2133 , 163., 163., 6.   , 97. , 5.89 ],
    'Y': [163.176  , 141., 141., 10.  , 63. , 5.64 ],
    'X': [142.67295, 134., 134., 4.5  , 88. , 5.45 ],
    'U': [168.064, 0., 0., 0  , 0. , 0. ],
    'O': [255.313, 0., 0., 0  , 0. , 0. ]
}

# sum up 3grams property
"""
sum_properties = property['A'] + property['B'] + property['C']
"""
def calculate_property(label):
    split_to_char = list(label)
    sum_properties = np.array([0., 0., 0., 0., 0., 0.])
    for char in split_to_char:
        sum_properties += np.array(pick_key(char))
    return sum_properties/3.


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

def pick_key(char):
    rand_dict = { 1 : 'N', 2 : 'D', 3 : 'E', 4 : 'Q', 5 : 'L', 6 : 'I'}
    try:
        return NGRAM_PROPERTIES[char]
    #return NGRAM_PROPERTIES[char]
    except:
        if char == 'B':
            return NGRAM_PROPERTIES[rand_dict[randint(1, 2)]]
        elif char == 'Z':
            return NGRAM_PROPERTIES[rand_dict[randint(3, 4)]]
        elif char == 'J':
            return NGRAM_PROPERTIES[rand_dict[randint(5, 6)]]

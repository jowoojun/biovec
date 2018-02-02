import numpy as np

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
    'Y': [1, 2, 3, 4, 5, 6]
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
pick up specific property
"""
def choose_category(sum_properties, category):
    if category == "mass":
        return sum_properties[0]

    elif category == "volume":
        return sum_properties[1]

    elif category == "van_der_waal":
        return sum_properties[2]

    elif category == "polarity":
        return sum_properties[3]

    elif category == "hydro":
        return sum_properties[4]

    elif category == "charge":
        return sum_properties[5]

"""
dictionary = {'ABC' : sum_properties}
"""
def make_property_dict(labels, category):
    property_dict = {}
    for label in labels:
        property_dict[label] = choose_category(calculate_property(label),
                                               category)
    return property_dict

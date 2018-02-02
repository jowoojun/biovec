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

def calculate_property(label):
    split_to_char = list(label)
    sum_properties = np.array([0, 0, 0, 0, 0, 0])
    for char in split_to_char:
        sum_properties += np.array(NGRAM_PROPERTIES[char])
    return sum_properties

def make_property_dict(labels):
    property_dict = {}
    for label in labels:
        property_dict[label] = calculate_property(label)
    return property_dict

def choose_category(sum_properties, category):
    if category is "mass":
        return sum_properties[0]

    elif category is "volume":
        return sum_properties[1]

    elif category is "van_der_waal":
        return sum_properties

    elif category is "polarity":
        return sum_properties

    elif category is "hydro":
        return sum_properties

    elif category is "charge":
        return sum_properties

    else:
        print "category is not on the list"

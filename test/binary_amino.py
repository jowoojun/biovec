'''
Binary representation of amino acid residue and amino acid sequence
e.g.
    'A' => [0, 0, 0, 0, 0]
    'AGGP' => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
'''

AMINO_ACID_BINARY_TABLE = {
    'A': [0, 0, 0, 0, 0],
    'C': [0, 0, 0, 0, 1],
    'D': [0, 0, 0, 1, 0],
    'E': [0, 0, 0, 1, 1],
    'F': [0, 0, 1, 0, 0],
    'G': [0, 0, 1, 0, 1],
    'H': [0, 0, 1, 1, 0],
    'I': [0, 0, 1, 1, 1],
    'K': [0, 1, 0, 0, 0],
    'L': [0, 1, 0, 0, 1],
    'M': [0, 1, 0, 1, 0],
    'N': [0, 1, 0, 1, 1],
    'P': [0, 1, 1, 0, 0],
    'Q': [0, 1, 1, 0, 1],
    'R': [0, 1, 1, 1, 1],
    'S': [1, 0, 0, 0, 0],
    'T': [1, 0, 0, 0, 1],
    'V': [1, 0, 0, 1, 0],
    'W': [1, 0, 0, 1, 1],
    'Y': [1, 0, 1, 0, 0]
}


def convert_amino_to_binary(amino):
    '''
    Convert amino acid to 1-dimentional 5 length binary array
    "A" => [0, 0, 0, 0, 0]
    '''
    if not AMINO_ACID_BINARY_TABLE.has_key(amino):
        return None
    return AMINO_ACID_BINARY_TABLE[amino]


def convert_amino_acid_sequence_to_vector(sequence):
    '''
    "AGGP" => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
    '''
    binary_vector = [convert_amino_to_binary(amino) for amino in sequence]
    if None in binary_vector:
        return None
    return binary_vector

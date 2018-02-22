from Bio import SeqIO
from collections import Counter
import gzip
import sys

class Pfam:
    def __init__(self):
        print("start")

    def pfam_parser(self, data_path):
        protein_family_dict = {}
        number_of_protein_in_family = {}
        print "\nMaking files about protein and family\n"

        #families = []
        with open(data_path) as f:
            for line in f:
                uniprot_name, family_name, vector_string = line.rstrip().split('\t', 2)
                #families.append(family_name)
                protein_family_dict[uniprot_name] = family_name
                if family_name in number_of_protein_in_family:
                    number_of_protein_in_family[family_name] += 1
                else:
                    number_of_protein_in_family[family_name] = 1
                
        return protein_family_dict, number_of_protein_in_family

if __name__ == '__main__':
    data_path = 'trained_models/protein_pfam_vector.csv'
    pf = Pfam()
    pfd, number_of_protein_in_family = pf.pfam_parser(data_path)

    with open('family_stat.txt', 'w') as outfile:
        for family, number_of_proteins in number_of_protein_in_family.iteritems():
            print ("a")
            outfile.write('{}\t{}\n'.format(family, number_of_proteins))


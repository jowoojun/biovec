from Bio import SeqIO
from collections import Counter
import gzip
import sys

class Pfam:
    def __init__(self):
        self.number_of_protein_in_family = Counter()

    def pfam_parser(self, data_path):
        protein_family_dict = {}
        print ("\nMaking files about protein and family\n")

        i = 0
        with gzip.open(data_path, 'rb') as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_name = record.name.split('/', 1)[0].lstrip('>')
                family_name = record.description.rsplit(';', 2)[1]
                protein_family_dict[uniprot_name] = family_name
                self.number_of_protein_in_family[family_name] += 1
                i = i+1
                print (i)


        return protein_family_dict, self.number_of_protein_in_family

if __name__ == '__main__':
    data_path = '../document/Pfam-A.fasta.gz'
    pf = Pfam()
    pf.pfam_parser(data_path)

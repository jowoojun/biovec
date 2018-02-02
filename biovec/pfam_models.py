from Bio import SeqIO
import gzip
import sys

class Pfam:
    def __init__(self, min_count = 20):
        self.min_count = min_count

    def pfam_parser(self, data_path):
        protein_family_dict = {}
        print "Making dictionary about protein and family\n"

        with gzip.open(data_path, 'rb') as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_name = record.name.split('/', 1)[0].lstrip('>')
                family_name = record.description.rsplit(';', 2)[1]
                protein_family_dict[uniprot_name] = family_name

        print protein_family_dict
        return protein_family_dict

if __name__ == '__main__':
    data_path = '../document/Pfam-A.fasta.gz'
    pf = Pfam()
    pf.pfam_parser(data_path)

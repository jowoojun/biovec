from Bio import SeqIO
from collections import Counter
import gzip
import sys

from Bio import SeqIO

class Pfam:
    def __init__(self):
        print("a")

    def pfam_parser(self, data_path):
        protein_family_dict = {}
        number_of_protein_in_family = {}
        print "\nMaking files about protein and family\n"

        with gzip.open(data_path, 'rb') as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                uniprot_name = record.name.split('/', 1)[0].lstrip('>')
                family_name = record.description.rsplit(';', 2)[1]
                protein_family_dict[uniprot_name] = family_name

                if family_name in number_of_protein_in_family:
                    number_of_protein_in_family[family_name] += 1
                else:
                    number_of_protein_in_family[family_name] = 1
        
        return protein_family_dict, number_of_protein_in_family

    def make_uniport_with_families(self):
        protein_families = {}
        protein_family_stat = Counter()
        with gzip.open('document/Pfam-A.fasta.gz', 'rb') as gzipped_file:
            for record in SeqIO.parse(gzipped_file, "fasta"):  
                # >F0S5U5_PSESL/156-195 F0S5U5.1 PF10417.8;1-cysPrx_C;    
                family_id = record.description.rsplit(';', 2)[-2]
                uniprot_id = record.name.split('/', 1)[0].lstrip('>') 
                protein_families[uniprot_id] = family_id
                #protein_family_stat[family_id] += 1

        #with open('family_stat.txt', 'w') as outfile:
        #    for family, number_of_proteins in protein_family_stat.iteritems():
        #        outfile.write('{}\t{}\n'.format(family, number_of_proteins))

        min_proteins_in_family = 20
        with gzip.open('../document/uniprot_sprot.fasta.gz', 'rb') as gzipped_file, \
        open("trained_models/uniprot_with_families.fasta", "w") as output_fasta:
            for record in SeqIO.parse(gzipped_file, "fasta"):
                uniprot_id = record.name.split('|')[2] 
                if uniprot_id in protein_families:
                    family = protein_families[uniprot_id]
                    #if protein_family_stat[family] >= min_proteins_in_family:
                    record.description += ' PFAM={}'.format(protein_families[uniprot_id])
                    SeqIO.write(record, output_fasta, "fasta")

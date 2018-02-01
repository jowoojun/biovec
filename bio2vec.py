from Bio import SeqIO

import biovec
import bio_tsne
import tensorflow as tf
import numpy as np
import os
import sys
import gzip

fasta_file = "document/uniprot_sprot.fasta.gz"
pv = biovec.ProtVec(fasta_file,
                    out="trained_models/ngram_corpus.txt")

print "Checking the file(trained_models/ngram_vector.csv)"
ngram_model_fname = "trained_models/ngram_vector.csv"
protein_model_fname = "trained_models/protein_vector.csv"

if not os.path.isfile(ngram_model_fname) or not os.path.isfile(protein_model_fname):
    print 'INFORM : There is no model file. Generate model file from data file...'
    pv.word2vec_init(ngram_model_fname)

    with gzip.open(fasta_file, 'rb') as fasta_file:
        with open(protein_model_fname, 'w') as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_name = record.name.split('|')[-1]
                protein_vector = pv.to_vecs(record.seq)
                output_file.write('{}\t{}\n'.format(protein_name, ' '.join(map(str, protein_vector))))
                sys.stdout.write(".")
else:
    print "INFORM : File's Existence is confirmed\n"
print "...OK\n"

print "Checking the file(trained_models/protein_pfam_vector.csv)"
protein_pfam_fasta_fname = "trained_models/protein_pfam_vector.fasta"
protein_pfam_model_fname = "trained_models/protein_pfam_vector.csv"

if not os.path.isfile(protein_pfam_fasta_fname) or not os.path.isfile(protein_pfam_model_fname):
    print 'INFORM : There is no pfam_model file. Generate pfam_model file from data file...'

    pf = biovec.Pfam(min_count=20)
    protein_family_dict = pf.pfam_parser("./document/Pfam-A.fasta.gz")

    with gzip.open(fasta_file, 'rb') as fasta_file:
        with open(protein_pfam_fasta_fname, 'w') as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_name = record.name.split('|')[2]
                if protein_name in protein_family_dict:
                    record.description += ' PFAM={}'.format(protein_family_dict[protein_name])
                    SeqIO.write(record, output_file, "fasta")
                sys.stdout.write(".")

    print "...OK\n"


    with open(protein_model_fname) as protein_vector_file:
        for line in protein_vector_file:
            uniprot_name, vector_string = line.rstrip.split('\t', 1)
            if uniprot_name in protein_family_dict:
                print('{}\t{}\t{}'.format(uniprot_name, protein_family_dict[uniprot_name], vector_string))

print "... Done\n"

from Bio import SeqIO

import biovec
import bio_tsne
import tensorflow as tf
import numpy as np
import os
import sys
import gzip

fasta_file = "./document/uniprot_sprot.fasta.gz"
pv = biovec.ProtVec(fasta_file,
                    out="./trained_models/ngram_corpus.txt")

print "Now we are checking the file(trained_models/ngram_vector.csv)"
ngram_model_fname = "./trained_models/ngram_vector.csv"
protein_model_fname = "./trained_models/protein_vector.csv"

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
    print "INFORM : File's Existence is confirmed"

print "... Done\n"



"""
model = biovec.models.load_protvec(model_fname)
print "Loading protvec"
print "... OK\n"

tsne = bio_tsne.BioTsne()
print "Making tsne"

tsne.make_tsne(model)
print "... OK\n"

print "Visualization"
tsne.visualization()
"""

from Bio import SeqIO
from theano import function, config, shared, tensor

from collections import Counter
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

print ("Checking the file(trained_models/ngram_vector.csv)")
ngram_model_fname = "trained_models/ngram_vector.csv"
protein_model_fname = "trained_models/protein_vector.csv"

model_ngram = "trained_models/ngram_model"

def open_gzip_fasta(fasta_file, protein_model_fname):
    with gzip.open(fasta_file, 'rb') as fasta_file:
        with open(protein_model_fname, 'w') as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_name = record.name.split('|')[-1]
                protein_vector = pv.to_vecs(record.seq, ngram_vectors)

                output_file.write('{}\t{}\n'.format(protein_name, ' '.join(map(str, protein_vector))))

def get_uniprot_protein_families(path):
    protein_families = {}
    protein_family_stat = Counter()
    for record in SeqIO.parse(path, "fasta"): 
        family_id = None
        for element in record.description.split():
            if element.startswith('PFAM'):
                family_id = element.split('=', 1)[1]
        if family_id:
            uniprot_id = record.name.split('|')[-1]
            protein_families[uniprot_id] = family_id
            protein_family_stat[family_id] += 1

    return protein_families, protein_family_stat



# uniport
if not os.path.isfile(ngram_model_fname) or not os.path.isfile(protein_model_fname):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(ngram_model_fname)
    pv.save(model_ngram)

    ngram_vectors = pv.get_ngram_vectors("trained_models/ngram_vector.csv")
    open_gzip_fasta(fasta_file, protein_model_fname)

else:
    print("INFORM : File's Existence is confirmed\n")

print ("...OK\n")


#================================================================================#

disprot_fasta = "document/disprot.fasta.gz"
dpv = biovec.ProtVec(disprot_fasta,
                     out="trained_models/disprot/disrprot_ngram_corpus.txt")

#print "Checking the file(trained_models/ngram_vector.csv)"

# disprot
disprot_ngram = "trained_models/disprot/disprot_ngram.csv"
disprot_protein = "trained_models/disprot/disprot_protein.csv"
if not os.path.isfile(disprot_ngram) or not os.path.isfile(disprot_protein):
    print 'INFORM : There is no vector model file. Generate model files from data file...'
    dpv.word2vec_init(disprot_ngram)
    #dpv.save(model_ngram)

    ngram_vectors = dpv.get_ngram_vectors(disprot_ngram)
    open_gzip_fasta(disprot_fasta, disprot_protein)

else:
    print "INFORM : File's Existence is confirmed\n"

print "...OK\n"


# pdb
pdb_fasta= "document/uniprot_sprot.fasta.gz"
ppv = biovec.ProtVec(fasta_file,out="trained_models/pdb/pdb_ngram_corpus.txt")

pdb_ngram = "trained_models/pdb/pdb_ngram.csv"
pdb_protein = "trained_models/pdb/pdb_protein.csv"
if not os.path.isfile(pdb_ngram) or not os.path.isfile(pdb_protein):
    print 'INFORM : There is no vector model file. Generate model files from data file...'
    ppv.word2vec_init(pdb_ngram)
    #pv.save(model_ngram)

    ngram_vectors = ppv.get_ngram_vectors(pdb_ngram)
    open_gzip_fasta(pdb_fasta, pdb_protein)

else:
    print "INFORM : File's Existence is confirmed\n"

print "...OK\n"

#===============================================================================#


print("Checking the file(trained_models/protein_pfam_vector.csv)")
protein_pfam_model_fname = "trained_models/protein_pfam_vector.csv"

if not os.path.isfile(protein_pfam_model_fname):
    print ('INFORM : There is no pfam_model file. Generate pfam_model files from data file...')

    min_proteins_in_family = 10
    pf = biovec.Pfam()
    uniprot_with_families = "trained_models/uniprot_with_families.fasta"
    pf.make_uniport_with_families()
    protein_families, protein_family_stat = get_uniprot_protein_families(uniprot_with_families)

    f = open(protein_pfam_model_fname, "w")
    with open(protein_model_fname) as protein_vector_file:
        for line in protein_vector_file:
            uniprot_name, vector_string = line.rstrip().split('\t', 1)
            if uniprot_name in protein_families:
                family = protein_families[uniprot_name]
                if protein_family_stat[family] >= min_proteins_in_family:
                    f.write('{}\t{}\t{}'.format(uniprot_name, protein_families[uniprot_name], vector_string) + "\n")
    f.close()

print ("... Done\n")

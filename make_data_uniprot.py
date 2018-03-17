from Bio import SeqIO
from theano import function, config, shared, tensor

from collections import Counter
import word2vec
import bio_tsne
import tensorflow as tf
import numpy as np
import os
import sys
import gzip


def make_protein_vector_for_uniprot(fasta_file, protein_vector_fname, ngram_vectors):
    with gzip.open(fasta_file, 'rb') as fasta_file:
        with open(protein_vector_fname, 'w') as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_name = record.name.split('|')[-1]
                protein_vector = pv.to_vecs(record.seq, ngram_vectors)

                output_file.write('{}\t{}\n'.format(protein_name, ' '.join(map(str, protein_vector))))

def make_protein_vector_for_other(fasta_file, protein_vector_fname, ngram_vectors):
    with gzip.open(fasta_file, 'rb') as fasta_file:
        with open(protein_vector_fname, 'w') as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_name = record.name.split(' ')[-1]
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

def make_uniport_with_families(Pfam_file, fasta_file, uniprot_with_families): 
    protein_families = {}
    protein_family_stat = Counter()
    with gzip.open(Pfam_file, 'rb') as gzipped_file:
        for record in SeqIO.parse(gzipped_file, "fasta"):  
            family_id = record.description.rsplit(';', 2)[-2]
            uniprot_id = record.name.split('/', 1)[0].lstrip('>') 
            protein_families[uniprot_id] = family_id

    with gzip.open(fasta_file, 'rb') as gzipped_file, open(uniprot_with_families, "w") as output_fasta:
        for record in SeqIO.parse(gzipped_file, "fasta"):
            uniprot_id = record.name.split('|')[2] 
            if uniprot_id in protein_families:
                family = protein_families[uniprot_id]
                record.description += ' PFAM={}'.format(protein_families[uniprot_id])
                SeqIO.write(record, output_fasta, "fasta")

def make_protein_pfam_vector_for_uniprot(protein_pfam_vector_fname, protein_vector_fname, protein_families, protein_family_stat):
    #Cut standard
    min_proteins_in_family = 100

    f = open(protein_pfam_vector_fname, "w")
    with open(protein_vector_fname) as protein_vector_file:
        for line in protein_vector_file:
            uniprot_name, vector_string = line.rstrip().split('\t', 1)
            if uniprot_name in protein_families:
                family = protein_families[uniprot_name]
                if protein_family_stat[family] >= min_proteins_in_family:
                    f.write('{},{},{}'.format(uniprot_name, protein_families[uniprot_name], vector_string) + "\n")
    f.close()


def make_protein_pfam_vector_for_other(protein_pfam_vector_fname, protein_vector_fname, fasta_file):
    #Cut standard
    min_proteins_in_family = 0

    protein_families = {}
    f = open(protein_pfam_vector_fname, "w")
    with open(protein_vector_fname) as protein_vector_file, gzip.open(fasta_file, 'rb') as gzipped_fasta:
        for record in SeqIO.parse(gzipped_fasta, "fasta"):
            gz_protein_name, gz_family = record.description.rstrip().split(' ', 1)
            print (gz_protein_name)
            print (gz_family)
            protein_families[gz_protein_name] = gz_family
        
        for line in protein_vector_file:
            protein_name, vector_string = line.rstrip().split('\t', 1)
            if protein_name in protein_families:
                family = protein_families[protein_name]
                f.write('{}\t{}\t{}'.format(protein_name, protein_families[protein_name], vector_string) + "\n")
                
    f.close()

fasta_file = "document/uniprot_sprot.fasta.gz"
Pfam_file = "document/Pfam-A.fasta.gz"
ngram_corpus_fname = "trained_models/ngram_vector.csv"
model_ngram = "trained_models/ngram_model"
protein_vector_fname = "trained_models/protein_vector.csv"
uniprot_with_families = "trained_models/uniprot_with_families.fasta"
protein_pfam_vector_fname = "trained_models/protein_pfam_vector.csv"

#Make corpus
pv = word2vec.ProtVec(fasta_file, out="trained_models/ngram_corpus.txt")

print ("Checking the file(trained_models/ngram_vector.csv)")
if not os.path.isfile(ngram_corpus_fname) or not os.path.isfile(protein_vector_fname):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    
    #Make ngram_vector.txt and word2vec model
    pv.word2vec_init(ngram_corpus_fname)
    pv.save(model_ngram) 

    #Get ngram and vectors
    ngram_vectors = pv.get_ngram_vectors(ngram_corpus_fname)
    
    #Make protein_vector.txt by ngram, vector, uniprot
    make_protein_vector_for_uniprot(fasta_file, protein_vector_fname, ngram_vectors)

else:
    print("INFORM : File's Existence is confirmed\n")

print ("...OK\n")


print("Checking the file(trained_models/protein_pfam_vector.csv)")
if not os.path.isfile(protein_pfam_vector_fname):
    print ('INFORM : There is no pfam_model file. Generate pfam_model files from data file...')
    
    #Make uniprot_with_family.fasta by uniprot, Pfam
    make_uniport_with_families(Pfam_file, fasta_file, uniprot_with_families)

    #Get protein_name, family_name, vectors
    protein_families, protein_family_stat = get_uniprot_protein_families(uniprot_with_families)

    #Make protein_pfam_vector_fname.csv by protein_name, family_name, vectors
    make_protein_pfam_vector_for_uniprot(protein_pfam_vector_fname, protein_vector_fname, protein_families, protein_family_stat)

print ("...Uniprot Done\n")

#===============================================================================#
# binary svm
# disprot
print("Start SVM dataset...\n")

directory = "trained_models/SVM_dataset"
if not os.path.exists(directory):
    os.makedirs(directory)
    print("directory(trained_models) created\n")

SVM_dataset_fasta = "document/dataset.fasta.gz"
dpv = word2vec.ProtVec(SVM_dataset_fasta,
                     out="trained_models/SVM_dataset/SVM_dataset_ngram_corpus.txt")

print ("Checking the file(trained_models/SVM_dataset/SVM_dataset_ngram.csv)")

SVM_ngram = "trained_models/SVM_dataset/SVM_dataset_ngram.csv"
SVM_protein = "trained_models/SVM_dataset/SVM_dataset_protein.csv"
#disprot_pfam_vector_fname = "trained_models/FG-NUPS/dis_pfam_vector.csv"
if not os.path.isfile(SVM_ngram) or not os.path.isfile(SVM_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    dpv.word2vec_init(SVM_ngram)

    ngram_vectors = dpv.get_ngram_vectors(SVM_ngram)
    make_protein_vector_for_other(SVM_dataset_fasta, SVM_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...SVM dataset Done\n")

#===============================================================================#
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("directory(trained_models) created\n")
        
# density map
# make directory
mkdir("trained_models/density_map")
mkdir("trained_models/density_map/dis-disprot")
mkdir("trained_models/density_map/disprot")
mkdir("trained_models/density_map/dis-fg-nups")
mkdir("trained_models/density_map/fg-nups")
mkdir("trained_models/density_map/pdb1")
mkdir("trained_models/density_map/pdb2")


# traing data path
dis_disprot = "./processed_data/density_map/dis-disprot.fasta.gz"
disprot     = "./processed_data/density_map/disprot.fasta.gz"
dis_fg_nups = "./processed_data/density_map/dis-fg-nups.fasta.gz"
fg_nups     = "./processed_data/density_map/fg-nups.fasta.gz"
pdb1        = "./processed_data/density_map/pdb1.fasta.gz"
pdb2        = "./processed_data/density_map/pdb2.fasta.gz"

# dis_disprot train
pv = word2vec.ProtVec(dis_disprot,
                     out="trained_models/density_map/dis-disprot/dis-disprot-ngram_corpus.txt")
print ("Checking the file dis-disprot")

dis_disprot_ngram   = "trained_models/density_map/dis-disprot/dis-disprot-ngram.csv"
dis_disprot_protein = "trained_models/density_map/dis-disprot/dis-disprot-protein.csv"
dis_disprot_model   = "trained_models/density_map/dis-disprot/dis-disprot-ngram-model"
if not os.path.isfile(dis_disprot_ngram) or not os.path.isfile(dis_disprot_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(dis_disprot_ngram)
    pv.save(dis_disprot_model) 

    ngram_vectors = pv.get_ngram_vectors(dis_disprot_ngram)
    make_protein_vector_for_other(dis_disprot, dis_disprot_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...dis-disprot Done\n")


# disprot train
pv = word2vec.ProtVec(disprot,
                     out="trained_models/density_map/disprot/disprot-ngram_corpus.txt")
print ("Checking the file disprot")

disprot_ngram   = "trained_models/density_map/disprot/disprot-ngram.csv"
disprot_protein = "trained_models/density_map/disprot/disprot-protein.csv"
disprot_model   = "trained_models/density_map/disprot/disprot-ngram-model"
if not os.path.isfile(disprot_ngram) or not os.path.isfile(disprot_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(disprot_ngram)
    pv.save(disprot_model) 

    ngram_vectors = pv.get_ngram_vectors(disprot_ngram)
    make_protein_vector_for_other(disprot, disprot_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...disprot Done\n")


# dis-fg-nups train
pv = word2vec.ProtVec(dis_fg_nups,
                     out="trained_models/density_map/dis-fg-nups/dis-fg-nups-ngram_corpus.txt")
print ("Checking the file dis-fg-nups")

dis_fg_nups_ngram   = "trained_models/density_map/dis-fg-nups/dis-fg-nups-ngram.csv"
dis_fg_nups_protein = "trained_models/density_map/dis-fg-nups/dis-fg-nups-protein.csv"
dis_fg_nups_model   = "trained_models/density_map/dis-fg-nups/dis-fg-nups-ngram-model"
if not os.path.isfile(dis_fg_nups_ngram) or not os.path.isfile(dis_fg_nups_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(dis_fg_nups_ngram)
    pv.save(dis_fg_nups_model) 

    ngram_vectors = pv.get_ngram_vectors(dis_fg_nups_ngram)
    make_protein_vector_for_other(dis_fg_nups, dis_fg_nups_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...dis_fg_nups Done\n")


# fg-nups train
pv = word2vec.ProtVec(fg_nups,
                     out="trained_models/density_map/fg-nups/fg-nups-ngram_corpus.txt")
print ("Checking the file fg-nups")

fg_nups_ngram   = "trained_models/density_map/fg-nups/fg-nups-ngram.csv"
fg_nups_protein = "trained_models/density_map/fg-nups/fg-nups-protein.csv"
fg_nups_model   = "trained_models/density_map/fg-nups/fg-nups-ngram-model"
if not os.path.isfile(fg_nups_ngram) or not os.path.isfile(fg_nups_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(fg_nups_ngram)
    pv.save(fg_nups_model) 

    ngram_vectors = pv.get_ngram_vectors(fg_nups_ngram)
    make_protein_vector_for_other(fg_nups, fg_nups_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...fg_nups Done\n")


# pdb1 train
pv = word2vec.ProtVec(pdb1,
                     out="trained_models/density_map/pdb1/pdb1-ngram_corpus.txt")
print ("Checking the file pdb1")

pdb1_ngram   = "trained_models/density_map/pdb1/pdb1-ngram.csv"
pdb1_protein = "trained_models/density_map/pdb1/pdb1-protein.csv"
pdb1_model   = "trained_models/density_map/pdb1/pdb1-ngram-model"
if not os.path.isfile(pdb1_ngram) or not os.path.isfile(pdb1_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(pdb1_ngram)
    pv.save(pdb1_model) 

    ngram_vectors = pv.get_ngram_vectors(pdb1_ngram)
    make_protein_vector_for_other(pdb1, pdb1_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...pdb1 Done\n")


# pdb2 train
pv = word2vec.ProtVec(pdb2,
                     out="trained_models/density_map/pdb2/pdb2-ngram_corpus.txt")
print ("Checking the file pdb2")

pdb2_ngram   = "trained_models/density_map/pdb2/pdb2-ngram.csv"
pdb2_protein = "trained_models/density_map/pdb2/pdb2-protein.csv"
pdb2_model   = "trained_models/density_map/pdb2/pdb2-ngram-model"
if not os.path.isfile(pdb2_ngram) or not os.path.isfile(pdb2_protein):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    pv.word2vec_init(pdb2_ngram)
    pv.save(pdb2_model) 

    ngram_vectors = pv.get_ngram_vectors(pdb2_ngram)
    make_protein_vector_for_other(pdb2, pdb2_protein,ngram_vectors)

else:
    print ("INFORM : File's Existence is confirmed\n")

print ("...OK\n")
print ("...pdb2 Done\n")


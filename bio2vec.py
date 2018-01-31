from Bio import SwissProt

import biovec
import bio_tsne
import tensorflow as tf
import numpy as np
import os

pv = biovec.ProtVec("./document/uniprot_sprot.fasta", out="./document/uniprot_sprot_corpus")

handle = open("./document/uniprot_sprot.dat")

print "Now we are checking the file(trained_models/2017trained_model)"
model_fname = "./trained_models/2017trained_model_protein"
model_3gram = "./trained_models/2017trained_model_3gram"
if not os.path.isfile(model_fname):
    print 'INFORM : There is no model file. Generate model file from data file...'
    pv.word2vec_init()
    pv["QAT"]
    for record in SwissProt.parse(handle):
        pv.save(model_3gram)
        pv.to_vecs(record.sequence)
        pv.save(model_fname)
else:
    print "INFORM : File's Existence is confirmed"

print "... OK\n"

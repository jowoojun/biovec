from Bio import SwissProt

import biovec
import bio_tsne
import tensorflow as tf
import numpy as np
import os

pv = biovec.ProtVec("./document/uniprot_sprot.fasta", out="./document/uniprot_sprot_corpus.txt")
pv["QAT"]

handle = open("./document/uniprot_sprot.dat")

print "Now we are checking the file(trained_models/2017trained_model"
if not os.path.isfile("./trained_models/2017trained_model"):
    for record in SwissProt.parse(handle):
        pv.to_vecs(record.sequence)
        pv.save('./trained_models/2017trained_model')

print "  OK"

model = biovec.models.load_protvec('./trained_models/2017trained_model')

print "Loading protvec"

print "Visualization"
tsne = bio_tsne.BioTsne()
tsne.visualization(model)

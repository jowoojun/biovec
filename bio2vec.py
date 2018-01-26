from Bio import SwissProt

import biovec
import bio_tsne
import tensorflow as tf
import numpy as np
import os

pv = biovec.ProtVec("./document/uniprot_sprot.fasta", out="./document/uniprot_sprot_corpus.txt")
pv["QAT"]

handle = open("./document/copy_uniprot_sprot.dat")

if os.path.isfile("./trained_models/trained_model"):
    for record in SwissProt.parse(handle):
        pv.to_vecs(record.sequence)
        pv.save('./trained_models/trained_model')

model = biovec.models.load_protvec('./trained_models/trained_model')

bio_tsne.visualization(model)

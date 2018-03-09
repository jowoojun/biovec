import bio_tsne.tsne_3gram as t3
import bio_tsne.tsne_protein as tp
import biovec
import ngrams_properties.ngrams_properties as pro
import pickle
import os
import numpy as np
print "PS , BSVM , CD"
str = raw_input()

if "PS"==str:
    
    print "Loading 3gram vector"
    model_3gram = "./trained_models/ngram_model"
    model = biovec.models.load_protvec(model_3gram)
    print "... Ok\n"

    print "Making tsne"
    tsne = t3.BioTsne()
    labels = model.wv.vocab.keys()
    #print labels
    property_list = pro.make_property_list(labels)
    tsne.make_tsne(model)
    f = open("./trained_models/ngram_2D_vector","rb")
    vectors = pickle.load(f)
    final_embedding = tsne.link_with_vector(vectors, property_list)
    print "... OK\n"

    print "Visualization"
    tsne.visualization(final_embedding)

elif "BSVM"==str:
    tsne = tp.BioTsne()
    
    # make disprot tsne
    dataset_2D  = "./trained_models/SVM_dataset/SVM_dataset_2D" 
    dataset_vec = "./trained_models/SVM_dataset/SVM_dataset_protein.csv" 
    if not os.path.isfile(dataset_2D):
        dataset_vectors = tsne.csv_to_array(dataset_vec)
        print len(dataset_vectors)
        tsne.make_tsne(dataset_2D ,dataset_vectors) 
elif "CD"==str:
    tsne = tp.BioTsne()

    # make `

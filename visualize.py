import bio_tsne.tsne_3gram as t3
import bio_tsne.tsne_protein as tp
import biovec
import ngrams_properties.ngrams_properties as pro
import pickle
import os
import numpy as np
import biovisual.bio_visual as bv
print "PS(protrin space) , BSVM(binay svm) , DM(density map)"
str = raw_input()

# make protein 100D-vec to 2D-vec
def protein_tsne(dataset_2D , dataset_vec):
    tsne = tp.BioTsne()
    if not os.path.isfile(dataset_2D):
        dataset_vectors = tsne.csv_to_array(dataset_vec)
        print len(dataset_vectors)
        tsne.make_tsne(dataset_2D ,dataset_vectors) 

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

elif "DM"==str:

    # Dis-FGNUP
    dis_fg_nups_2D  = "./trained_models/density_map/dis-fg-nups/dis-fg-nups-2D" 
    dis_fg_nups_vec = "./trained_models/density_map/dis-fg-nups/dis-fg-nups-protein.csv" 
    protein_tsne(dis_fg_nups_2D , dis_fg_nups_vec)

    # FGNUPS
    fg_nups_2D  = "./trained_models/density_map/fg-nups/fg-nups-2D" 
    fg_nups_vec = "./trained_models/density_map/fg-nups/fg-nups-protein.csv" 
    protein_tsne(fg_nups_2D , fg_nups_vec)


    # PDB random1
    pdb1_2D  = "./trained_models/density_map/pdb1/pdb1-2D" 
    pdb1_vec = "./trained_models/density_map/pdb1/pdb1-protein.csv" 
    protein_tsne(pdb1_2D , pdb1_vec)

    # PDB random2
    pdb2_2D  = "./trained_models/density_map/pdb2/pdb2-2D" 
    pdb2_vec = "./trained_models/density_map/pdb2/pdb2-protein.csv" 
    protein_tsne(pdb2_2D , pdb2_vec)

    # Dis-Disprot
    dis_disprot_2D  = "./trained_models/density_map/dis-disprot/dis-disprot-2D" 
    dis_disprot_vec = "./trained_models/density_map/dis-disprot/dis-disprot-protein.csv" 
    protein_tsne(dis_disprot_2D , dis_disprot_vec)

    # Disprot
    disprot_2D  = "./trained_models/density_map/disprot/disprot-2D" 
    disprot_vec = "./trained_models/density_map/disprot/disprot-protein.csv" 
    protein_tsne(disprot_2D , disprot_vec)

    visual = bv.BioVisual()
    visual.visual_vec(dis_disprot_2D , disprot_2D ,dis_fg_nups_2D , fg_nups_2D , pdb1_2D , pdb2_2D)

import bio_tsne.tsne_3gram as t3
import bio_tsne.tsne_protein as tp
import biovec
import ngrams_properties.ngrams_properties as pro
import pickle

print "3gram or protein"
str = raw_input()

if "3gram"==str:
    
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
    f = open("./trained_models/2D_vec/ngram_2D_vector","rb")
    vectors = pickle.load(f)
    final_embedding = tsne.link_with_vector(vectors, property_list)
    print "... OK\n"

    print "Visualization"
    tsne.visualization(final_embedding)

else :
    tsne = tp.BioTsne()
    tsne.make_tsne() 
    tsne.visualization()

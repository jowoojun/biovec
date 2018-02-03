import bio_tsne.tsne_3gram as t3
import bio_tsne.tsne_protein as tp
import biovec
from ngrams_properties import ngrams_properties

print "3gram or protein"
str = raw_input()

if "3gram"==str:
    model_3gram = "./trained_models/ngram_model"
    model = biovec.models.load_protvec(model_3gram)

    tsne = t3.BioTsne()

else :
    model_fname = "./trained_models/2017trained_model_protein"
    model = biovec.models.load_protvec(model_fname)\

    tsne = tp.BioTsne()

print "Loading protvec"
print "... OK\n"


print "Making tsne"
labels = model.keys()
property_dict = make_property_dict(labels, "mass")
tsne_protein = tsne.make_tsne(model)
final_embedding = link_with_vector(vectors, property_dict)

print "... OK\n"


print "Visualization"
tsne.visualization(final_embedding)

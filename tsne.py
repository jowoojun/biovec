import bio_tsne.tsne_3gram as bt
import biovec

#model_fname = "./trained_models/2017trained_model_protein"
model_3gram = "./trained_models/ngram_model"

#model_protein = biovec.models.load_protvec(model_fname)
model_3gram = biovec.models.load_protvec(model_3gram)

print "Loading protvec"
print "... OK\n"

tsne = bt.BioTsne()

print "Making tsne"
tsne_protein = tsne.make_tsne(model_3gram )
print "... OK\n"

print "Visualization"
tsne.visualization()

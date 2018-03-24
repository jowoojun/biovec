# 2017Bio2Vec
Protein classification over sum of protein ngrams vector representation

Ordinarily, biological information is represented by an array of characters, but it is suggested that by expressing it as a vector, information can be stored more easily for analysis. As a specific application range,

1. family classification
2. protein visualization
3. structure prediction
4. disordered protein identification
5. protein-protein interaction prediction.

Such Classification and prediction are easy to understand usage, but personally I felt that protein visualization would be most useful. Unless the sequence is short or the structure is already known, it seems that the current method of grasping the whole of protein is not popular in general, so I think that such expression method has certain usefulness. Although this idea seems strange at first glance, it is recognized to some extent in natural language.

See another implementation in https://github.com/kyu999/biovec, https://github.com/peter-volkov/biovec

#If you don't have Database, you can download from the below link.

Uniprot (Swiss-prot)
 - http://www.uniprot.org/downloads

Disprot
 - http://www.disprot.org/browse

#If you don't working on mac OS try this
 - https://github.com/tensorflow/tensorflow/issues/5089


#How to install and use
1. Install python packages.
  - pip install -r requirements.txt

  cf) If you use macos and get a problem about installation issue with matplotlib python, go to the next link.
     https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

2. Download data file.
  - If you want to just test this program, you can download small database from below link.
      https://drive.google.com/open?id=1-hHiEMMPlNM4XLY2i3guaI16gLuW-VRc

  - On the other hand, If you want to run original program, you have to download original database from below link.
       https://drive.google.com/open?id=1gtzoPyNePFW8RcuUmCh-JkVW4e7sc0gh

3. Move the downloaded file to our project directory.

4. And then, unzip downloaded file.
  1) If you download small DB
  - tar -xzvf small_DB.tar.gz

  2) If you download original DB
  - tar -xzvf original_DB.tar.gz

5. Run make_data_uniprot.py

6. Now you get ngram's corpus and ngram's vectors, protein's vectors, protein's families to uniprot_sprot.fasta

7. If you want to get how to we classify proteins into each family, please run bio_svm/train_svm_biovec.py
  - then you want to know how to we organize SVM using RBF kernels, try next commend.
		tensorboard --logdir=./logs


description 
  - word2vec : Generating word2vec model from protein databases(gensim).

  - document : Protain databases(uniprot, Pfam, disprot, PDB...).

  - bio_tsne : Visualization protain vectors(TSNE)

  - trained_models : Proprocessed data made by make_data_uniprot.py

  - bio_svm : Classifying proteins into protein's families


#How to run tsne.py

1. Install python packages.
  - pip install -r requirements.txt

2. run bio2vec
  - python bio2vec.py

3. run tsne
  - python tsne.py

4. choose 3gram ro protein
   just type what you want

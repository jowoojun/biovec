# 2017Bio2Vec

If you don't have Database, you can download from the below link.
=================================================================

Uniprot (Swiss-prot)
 - http://www.uniprot.org/downloads

Disprot
 - http://www.disprot.org/browse

If you don't working on mac OS try this
=======================================
 - https://github.com/tensorflow/tensorflow/issues/5089


How to install
=======================================
1. Install python packages.
  - pip install -r requirements.txt

  cf) If you use macos and get a problem about installation issue with matplotlib python, go to the next link.
     https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

2. Download data file.
  1) If you want to just test this program, you can download small database from below link.
  - https://drive.google.com/open?id=1-hHiEMMPlNM4XLY2i3guaI16gLuW-VRc

  2) On the other hand, If you want to run original program, you have to download original database from below link.
  - https://drive.google.com/open?id=1gtzoPyNePFW8RcuUmCh-JkVW4e7sc0gh

3. Move the downloaded file to a directory obtained from git clone.

4. Then, unzip data file.
  1) If you download small DB
  - tar -xzvf small_DB.tar.gz

  2) If you download original DB
  - tar -xzvf original_DB.tar.gz


 description 
  - biovec : Generating word2vec model(gensim).

  - document : Protain database(uniprot, trained model).

  - bio_tsne : Visualization protain vectors(TSNE)

  - test, trained_models : Just for referances.




How to run  tsne.py
============

1. Install python packages.
  - pip install -r requirements.txt

2. run bio2vec
  - python bio2vec.py

3. run tsne
  - python tsne.py

4. choose 3gram ro protein
   just type what you want

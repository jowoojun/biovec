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
  - https://drive.google.com/open?id=14zxmM1XtYFZ319OS4-1op-4zRq-0MGRY 
  
  2) On the other hand, If you want to run original program, you have to download original database from below link.
  - https://drive.google.com/open?id=12BU8aFtWFtA5x9iTIV4TsxCpniT9Neiu

3. Move the downloaded file to a directory obtained from git clone.

4. Then, unzip data file.
  - tar -xzvf document.tar.gz


 description 
  - biovec : Generating word2vec model(gensim).

  - document : Protain database(uniprot, trained model).

  - bio_tsne : Visualization protain vectors(TSNE)

  - test, trained_models : Just for referances.

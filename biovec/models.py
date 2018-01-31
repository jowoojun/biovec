from Bio import SeqIO
from gensim.models import word2vec

import sys
import os

"""
 'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
"""
def split_ngrams(seq, n):
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams


'''
Args:
    corpus_fname: corpus file name
    n: the number of chunks to split. In other words, "n" for "n-gram"
    out: output corpus file path
Description:
    Protvec uses word2vec inside, and it requires to load corpus file
    to generate corpus.
'''
def generate_corpusfile(corpus_fname, n, out):
    f = open(out, "w")
    for r in SeqIO.parse(corpus_fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
        sys.stdout.write(".")

    f.close()


def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):

    """
    Either fname or corpus is required.

	corpus_fname: fasta file for corpus
    corpus: corpus object implemented by gensim
    n: n of n-gram
    out: corpus output file path
    min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
    """
    def __init__(self, corpus_fname=None, corpus=None, n=3, size=100,
                 out="corpus.txt",  sg=1, window=25, min_count=2, workers=3):
        skip_gram = True

        self.n = n
        self.size = size
        self.corpus_fname = corpus_fname
        self.sg = int(skip_gram)
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.out = out

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if corpus_fname is not None:
            print 'Now we are checking whether corpus file exist'
            if not os.path.isfile(out):
                print 'INFORM : There is no corpus file. Generate Corpus file from fasta file...'
                generate_corpusfile(corpus_fname, n, out)
            else:
                print "INFORM : File's Existence is confirmed"
            self.corpus = word2vec.Text8Corpus(out)
            print "\n... OK\n"

    def word2vec_init(self):
        word2vec.Word2Vec.__init__(self, self.corpus, size=self.size, sg=self.sg, window=self.window, min_count=self.min_count, workers=self.workers)
        Word2Vec([line.rstrip().split() for line in open(out)], min_count =
                 self.min_count, size=self.size, sg=self.sg,
                 window=self.window)
        model.wv.save_word2vec_format('{}_vector.txt'.format(out))


    """
    convert sequence to three n-length vectors
    e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
    """
    def to_vecs(self, seq):
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs

from Bio import SwissProt

import biovec
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
'''
handle = open("copy_uniprot_sprot.dat")

#read a data
def read_data(handle):
    sequences = list()
	
    for record in SwissProt.parse(handle):
        s = record.sequence[0:]
        s_len = len(s)
        a = 0
        while a < s_len/3:
            sequences.append(s[a*3:(a+1)*3])
            a += 1

        s = record.sequence[1:]
        s_len = len(s)
        a = 0
        while a < s_len/3:
            sequences.append(s[a*3:(a+1)*3])
            a += 1

        s = record.sequence[2:]
        s_len = len(s)
        a = 0
        while a < s_len/3:
            sequences.append(s[a*3:(a+1)*3])
            a += 1
    return sequences
'''

pv = biovec.ProtVec("uniprot_sprot.fasta", out="uniprot_sprot_corpus.txt")
pv["QAT"]


handle = open("copy_uniprot_sprot.dat")

for record in SwissProt.parse(handle):
	pv.to_vecs(record.sequence)
	pv.save('models_path')

pv2 = biovec.models.load_protvec('models_path')

"""
print('Reading...')
word_sequence = read_data(handle)
word_list = word_sequence[:]
del handle
print('Data size', len(word_sequence))

#word_sequence = " ".join(sentences).split()
#word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

skip_grams = []

for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word
	return random_inputs, random_labels

training_epoch = 300
learning_rate = 0.1
batch_size = 20
embedding_size = 2
num_sampled = 15
voc_size = len(word_list)

inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    trained_embeddings = embeddings.eval()


for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()
"""

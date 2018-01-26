# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

font_name = matplotlib.font_manager.FontProperties(
		fname="/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf", size=18
		).get_name()
matplotlib.rc('font', family=font_name)

contexts = ["나 강아지 좋다", "나 고양이 좋다", "나 동물 좋다", "강아지 동물 좋다", "여자친구 고양이 강아지 좋다", "강아지 우유 고양이 생선 좋다", "나 남자친구 싫다", "나 게임 코딩 좋다",
		 "공부 영어 싫다", "우유 여자친구 좋다", "나 영화 게임 만화 좋다" , "나 영화 강아지 좋다", "나 남자친구 우유 싫다", "나 코딩 생선 좋다", "강아지 고양이 영어 좋다", "나 우유 게임 만화 좋다",
		 "치킨 좋다", "치킨 좋다", "치킨 좋다", "치킨 좋다", "치킨 좋다", "치킨 좋다"]

def build_dataset(contexts):
	word_list = " ".join(contexts).split()
	word_list = list(set(word_list))
	
	word_dict = {w:i for i, w in enumerate(word_list)}
	word_index = [word_dict[word] for word in word_list]

	words = list()
	for word in word_list:
		words.append(word)

	return words, word_index

word_list, word_index = build_dataset(contexts)

def make_skip_grams(word_list):
	skip_grams = []
	for i in range(1, len(word_list) - 1):
		target = word_list[i]
		word_window = [word_list[i-1], word_list[i+1]]

		for w in word_window:
			skip_grams.append([target, w])

	return skip_grams

skip_grams = make_skip_grams(word_index)

def random_batch(data, size):
	random_inputs = []
	random_labels = []
	random_index = np.random.choice(range(len(data)), size, replace=False)

	for i in random_index:
		random_inputs.append(data[i][0])
		random_labels.append([data[i][1]])
	
	return random_inputs, random_labels



training_times = 10000
batch_size = 20
learning_rate = 0.1
embedding_size = 2
sample = 15
voc_size = len(word_list)


train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

nce_weights = tf.Variable(tf.truncated_normal([voc_size, embedding_size],
			                        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(
			weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=sample, num_classes=voc_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(1, training_times + 1):
	batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
	feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
	_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)

	if step % 10 == 0:
		print("current step ",step, ": ",loss_val)

training_embeddings = sess.run(embeddings)

for i, label in enumerate(word_list):
	x, y = training_embeddings[i]
	plt.scatter(x,y)
	plt.annotate(label, xy = (x,y), xytext=(5,2),
			textcoords='offset points',ha='right', va='bottom')
plt.show()

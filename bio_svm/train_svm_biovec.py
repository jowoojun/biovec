import argparse
import sys
import os

import numpy as np

import tensorflow as tf
import pandas
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from collections import Counter

ops.reset_default_graph()

def get_data(sess, path):
    print("Read_csv...")
    dataframe = pandas.read_csv(path, header=None)
    dataset = dataframe.values
    family = dataset[:,1]
    vectors = dataset[:,2:].astype(float)
    data_size = len(family)
    print("Done...\n")

    # Sampling data  = 40%
    print("Data sampling...")
    sample_indices = np.random.choice(len(vectors), int(round(len(vectors)*1.0)), replace=False)
    vectors_sample = vectors[sample_indices]
    family_sample = family[sample_indices]
    print("Done...\n")

    #vectors_else, family_else = None, None

    print("Labeling...")
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(family_sample)
    families_encoded = np.array(label_encoder.transform(family_sample), dtype=np.int32)
    family_sample = None
    depth = families_encoded.max()
    print("Done...\n")

    print("One hot Encoding...")
    tf_onehot = tf.one_hot(families_encoded, depth, on_value=1.0, off_value=0.0)
    np_onehot = tf_onehot.eval(session=sess)
    print("Done...\n")

    min_on_training = vectors_sample.min(axis=0)
    range_on_training = (vectors_sample - min_on_training).max(axis=0)
    

    vectors_train_scaled = (vectors_sample - min_on_training) / range_on_training
    
    return label_encoder, vectors_train_scaled, np_onehot, depth, data_size

def save_model_metrics(model_params_string, families_test, predicted_families, label_encoder):
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))

        test_predictions = predicted_families
        prediction_counter = Counter()
        for index, predicted_family in enumerate(predicted_families):
            predicted_family = label_encoder.inverse_transform(predicted_family.astype('int64'))
            actual_family = label_encoder.inverse_transform(families_test[index].astype('int64'))
            prediction_counter[actual_family==predicted_family] += 1
            outfile.write('actual_family={} predicted_family={} correct={}\n'.format(
                                                                                     actual_family, 
                                                                                     predicted_family, 
                                                                                     actual_family==predicted_family))
        tp_rate = float(prediction_counter[True]) / sum(prediction_counter.values())
        outfile.write('counter = {} TP_rate = {}\n'.format(prediction_counter, tp_rate))
    

def main():
    parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
    parser.add_argument('--sample', type=str, default='../trained_models/protein_pfam_vector.csv')
    args = parser.parse_args()

    sess = tf.Session()

    print ("Start getting data...")
    label_encoder, x_vals, y_vals, depth, data_size = get_data(sess, args.sample)
    print ("Done...\n")

    batch_size = 100
    learning_rate = 0.01
    training_epochs = 5

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 100], dtype=tf.float32)
    y_target = tf.placeholder(shape=[depth, None], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, 100], dtype=tf.float32)

    # Create variables for svm
    b = tf.Variable(tf.random_normal(shape=[depth, batch_size]))


    # Gaussian (RBF) kernel
    gamma = tf.constant(-5.0)
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1,1])
    sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Declare function to do reshape/batch multiplication
    def reshape_matmul(mat):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [depth, batch_size, 1])
        return(tf.matmul(v2, v1))

    # Compute SVM Model
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = reshape_matmul(y_target)

    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
    loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
    prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = my_opt.minimize(loss)

    # Initialize variables
    model_path = "../trained_models/svm.ckpt"
    init = tf.global_variables_initializer()

    #b_summary = tf.summary.scalar('b', b)
    loss_summary = tf.summary.scalar('loss', loss)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    
    sess.run(init)
    saver = tf.train.Saver()
    
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    # Training loop
    loss_vec = []
    batch_accuracy = []
    y_vals = np.transpose(y_vals)
    total_batch = int(data_size / batch_size)
    for i in range(total_batch):
        #rand_index = np.random.choice(len(x_vals), size=batch_size, replace=False)
        #rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_index = [i for i in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = x_vals[rand_index]
        rand_y = y_vals[:,rand_index]

        sess.run(optimizer, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x})
        batch_accuracy.append(acc_temp)

        summary = sess.run(merged_summary, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
        summary_writer.add_summary(summary, i)
        
        #if (i+1)==0:
        print('Step #' + str(i+1))
        print(',Loss = ' + str(temp_loss))
        print(',accuracy = ' + str(acc_temp)) 
        print('\n')


    print ('total accuracy = ' + str(tf.reduce_mean(batch_accuracy).eval(session=sess)))
    save_path = saver.save(sess, model_path)
    print ("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    main()


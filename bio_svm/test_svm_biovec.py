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
    print("Done...\n")

    # Sampling data  = 30%
    print("Data sampling...")
    sample_indices = np.random.choice(len(vectors), int(round(len(vectors)*0.99)), replace=False)
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
    
    return label_encoder, vectors_train_scaled, np_onehot, depth

def save_model_metrics(model_params_string, families_test, predicted_families, label_encoder):
    actual_family_and_num = dict()
    predicted_family_and_num = dict()
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))

        test_predictions = predicted_families
        prediction_counter = Counter()

        for index, predicted_family in enumerate(predicted_families):
            predicted_family = label_encoder.inverse_transform(predicted_family.astype('int64'))
            actual_family = label_encoder.inverse_transform(families_test[index].astype('int64'))
            prediction_counter[actual_family==predicted_family] += 1
            if actual_family in actual_family_and_num:
                actual_family_and_num[actual_family] += 1
            else:
                actual_family_and_num[actual_family] = 1

            if predicted_family == actual_family:
                if predicted_family in predicted_family_and_num:
                    predicted_family_and_num[predicted_family] += 1
                else:
                    predicted_family_and_num[predicted_family] = 1

        for x in actual_family_and_num:
            actual = actual_family_and_num[x]
            if not x in predicted_family_and_num:
                predicted_family_and_num[x] = 0
            predicted = predicted_family_and_num[x]
            acc = float(predicted) / float(actual)
            outfile.write('{}\t\t{}\t\t{}\t\t{:f}\n'.format(x, actual, predicted, acc))
            #outfile.write('actual_family={} predicted_family={} correct={}\n'.format(actual_family, predicted_family, actual_family==predicted_family))

        tp_rate = float(prediction_counter[True]) / sum(prediction_counter.values())
        outfile.write('counter = {} TP_rate = {}\n'.format(prediction_counter, tp_rate))
    
def main():
    parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
    parser.add_argument('--sample', type=str, default='../trained_models/protein_pfam_vector.csv')
    args = parser.parse_args()

    sess = tf.Session()

    print ("Start getting data...")
    label_encoder, x_test, y_test, num_of_families = get_data(sess, args.sample)
    print ("Done...\n")
    depth = 23
    batch_size = 100

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
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # Initialize variables
    save_path = "../trained_models/svm.ckpt"
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    # test_model
    sess.run(init)
    saver.restore(sess, save_path)
    print ("Model restored from file: %s" %save_path)

    y_test = np.transpose(y_test)
    used_test_y = np.zeros(shape=(0))
    predicted = np.zeros(shape=(0))

    for i in range(100):
        #rand_index = np.random.choice(len(x_test), size=batch_size, replace=False)
        rand_index = np.random.choice(len(x_test), size=batch_size)
        rand_x = x_test[rand_index]
        rand_y = y_test[:,rand_index]
        if depth != num_of_families:
            redundancy = [[0 for col in range(batch_size)] for row in range(depth-num_of_families)]
            rand_y = np.concatenate([rand_y, redundancy])
       
        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                                 y_target: rand_y,
                                                 prediction_grid:rand_x})

        predicted_families = sess.run(prediction, feed_dict={x_data: rand_x, 
                                                             y_target: rand_y, 
                                                             prediction_grid:rand_x})
        rand_y = tf.argmax(rand_y, 0)
        rand_y = rand_y.eval(session=sess)

        used_test_y = np.append(used_test_y, rand_y)
        predicted = np.append(predicted, predicted_families)
        if (i+1)%25==0:
            print('Step #' + str(i+1))
            print('accuracy = ' + str(acc_temp)) 
            print('\n')

    save_model_metrics("rbf_model",  used_test_y, predicted, label_encoder)

if __name__=='__main__':
    main()

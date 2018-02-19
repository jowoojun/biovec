import argparse
import sys
import os

import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib.learn.python.learn.estimators import estimator
ops.reset_default_graph()

def get_data(sess, path):
    families_str = []
    vectors_float = []
    with open(path) as protein_pfam:
        for line in protein_pfam:
            uniprot_id, family, vector = line.rstrip().split('\t', 2)
            families_str.append(family)
            vectors_float.append(np.array(map(float, vector.split()), dtype=np.float32))

    vectors_array = np.array(vectors_float)
    vectors_float = None

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(families_str)
    families_encoded = np.array(label_encoder.transform(families_str), dtype=np.int32)
    families_str = None
    print "a"
    depth = families_encoded.max()+1
    tf_onehot = tf.one_hot(families_encoded, depth, on_value=1.0, off_value=0.0)
    np_onehot = tf_onehot.eval(session=sess)
    print "b"

    vectors_train, vectors_test, families_train, families_test = train_test_split(vectors_array, families_encoded, test_size=0.25, random_state=1)
    print "c"
    depth = families_encoded.max()

    vectors_array, families_encoded, families_binary_labels = None, None, None
    print "d"

    min_on_training = vectors_train.min(axis=0)
    range_on_training = (vectors_train - min_on_training).max(axis=0)
    print "e"

    vectors_train_scaled = (vectors_train - min_on_training) / range_on_training
    vectors_test_scaled = (vectors_test - min_on_training) / range_on_training
    print "f"

    return label_encoder, vectors_train_scaled, vectors_test_scaled, families_train, families_test, depth

    #y_vals = np.array(tf.one_hot(families_encoded, depth, off_value=-1))
    
    #for i in range(0, 7200):
    #    val = np.array([1 if y==i else -1 for y in families_encoded])
    #    y_vals.append(val)
    #    i = i + 1

    #b = np.zeros((families_encoded.size, families_encoded.max() + 1))
    #b[np.arange(families_encoded.size), families_encoded] = 1


def save_model_metrics(model_params_string, vectors_test, families_test, predicted_families, label_encoder):
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        #outfile.write('score: {}\n'.format(model.score(vectors_test, families_test)))
        #print('cross_val_test', cross_val_score(model, vectors_test, families_test, scoring='neg_log_loss'))
        #print('cross_val_train', cross_val_score(model, vectors_train, families_train, scoring='neg_log_loss'))
        #outfile.write('f1_macro: {}\n'.format(f1_score(families_test, predicted_families, average='macro')))
        #outfile.write('f1_micro: {}\n'.format(f1_score(families_test, predicted_families, average='micro')))
        #outfile.write('f1_weighted: {}\n'.format(f1_score(families_test, predicted_families, average='weighted')))
        families_test = tf.argmax(families_test,0)
        predicted_families = tf.argmax(predicted_families, 0)
        outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))

        test_predictions = predicted_families
        prediction_counter = Counter()
        for index, predicted_family in enumerate(predicted_families):
            predicted_family = label_encoder.inverse_transform(predicted_family)
            actual_family = label_encoder.inverse_transform(families_test[index])
            prediction_counter[actual_family==predicted_family] += 1
            outfile.write('actual_family={} predicted_family={} correct={}\n'.format(
                                                                                     actual_family, 
                                                                                     predicted_family, 
                                                                                     actual_family==predicted_family))
        tp_rate = float(prediction_counter[True]) / sum(prediction_counter.values())
        outfile.write('counter = {} TP_rate = {}\n'.format(prediction_counter, tp_rate))
    

def main():
    parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
    parser.add_argument('--sample', type=str, default='./trained_models/protein_pfam_vector.csv')
    #parser.add_argument('--type', type=str, default='svc_linear')
    args = parser.parse_args()

    sess = tf.Session()

    print "Start getting data..."
    label_encoder, x_vals, x_test, y_vals, y_test, depth = get_data(sess, args.sample)
    print "Done...\n"
    print x_vals.shape
    print x_test.shape
    print y_vals.shape
    print y_test.shape
    print depth
    batch_size = 100

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 100], dtype=tf.float32)
    y_target = tf.placeholder(shape=[depth, None], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, 100], dtype=tf.float32)

    # Create variables for svm
    b = tf.Variable(tf.random_normal(shape=[depth,batch_size]))


    # Gaussian (RBF) kernel
    gamma = tf.constant(-10.0)
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
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    loss_vec = []
    batch_accuracy = []
    for i in range(100):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[:,rand_index]
        
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:rand_x})


        batch_accuracy.append(acc_temp)

        if (i+1)%25==0:
            print('Step #' + str(i+1))
            print('Loss = ' + str(temp_loss))
            print('accuracy,' + str(acc_temp)) 
    
    #predicted_families = sess.run(prediction, feed_dict={x_data: x_test, y_target: y_test, prediction_grid:x_test})
    #print prediction_grid.shape
    #print y_test.shape
    #save_model_metrics("rbf_model", x_test, y_test, predicted_families, label_encoder)
    
"""
    # Create a mesh to plot points in
    x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
    y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.01))
    print type(xx.ravel())
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    ravel = []
    print type(ravel)
    for i in range(0, 100):
        x_min, x_max = x_vals[:, i].min() - 1, x_vals[:, i].max() + 1
        xx = np.array[np.meshgrid(np.arange(x_min, x_max, 0.02))]
        print xx.ravel()
        print type(xx.ravel())
        i = i + 1

    grid_points = np.c_[ravel]
    
    print rand_x.shape
    print rand_y.shape
    print grid_points.shape
    
    grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
                                                       y_target: rand_y,
                                                       prediction_grid: grid_points})
    grid_predictions = grid_predictions.reshape(xx.shape)

    print grid_points
    print ("grid_predict: {}".format(grid_predictions))
"""

if __name__ == '__main__':
    main()


import argparse
import numpy as np

import tensorflow as tf
import pandas
from scipy.sparse import csc_matrix
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.python.framework import ops
from collections import Counter

ops.reset_default_graph()

def get_data(sess, path):
    print("Read_csv...")
    dataframe = pandas.read_csv(path, header=None)
    dataset = dataframe.values
    family = dataset[:,1]
    vectors = dataset[:,2:].astype(float)
    print("Done...\n")

    print("Labeling...")
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(family)
    families_encoded = np.array(label_encoder.transform(family), dtype=np.int32)
    family = None
    depth = families_encoded.max() + 1
    print("Done...\n")

    print("One hot Encoding...")
    rows = np.arange(families_encoded.size)
    cols = families_encoded
    data = np.ones(families_encoded.size)
    np_onehot = csc_matrix((data, (rows, cols)), shape=(families_encoded.size, families_encoded.max()+1))
    print("Done...\n")


    min_on_training = vectors.min(axis=0)
    range_on_training = (vectors - min_on_training).max(axis=0)
    

    vectors_train_scaled = (vectors - min_on_training) / range_on_training
    
    return label_encoder, vectors_train_scaled, np_onehot, depth

def save_model_metrics(model_params_string, families_test, predicted_families, label_encoder):
    actual_family_and_num = dict()
    predicted_family_and_num = dict()
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))
        confusion = metrics.confusion_matrix(families_test, predicted_families)
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

        for index, actual_family in enumerate(actual_family_and_num):
            actual = actual_family_and_num[actual_family]
            if not actual_family in predicted_family_and_num:
                predicted_family_and_num[actual_family] = 0
            predicted = predicted_family_and_num[actual_family]

            TP = confusion[index, index]
            FP = np.sum(confusion[index,:]) - TP
            FN = np.sum(confusion[:,index]) - TP
            TN = np.sum(confusion) - FN - FP - TP

            acc = float(predicted) / float(actual)
            acc_temp = float(TP + TN)/ float(TP + FP + TN + FN)
            sensitivity = float(TP) / float(TP + FP)
            specificity = float(TN) / float(FN + TN)

            outfile.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(actual_family, actual, predicted, acc, sensitivity, specificity, acc_temp))
            #outfile.write('actual_family={} predicted_family={} correct={}\n'.format(actual_family, predicted_family, actual_family==predicted_family))

        tp_rate = float(prediction_counter[True]) / sum(prediction_counter.values())
        outfile.write('counter = {} TP_rate = {}\n'.format(prediction_counter, tp_rate))
    
def main():
    parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
    parser.add_argument('--sample', type=str, default='../trained_models/protein_pfam_vector.csv')
    args = parser.parse_args()

    sess = tf.Session()

    print ("Start getting data...")
    label_encoder, x_test, y_test_sparse, num_of_families = get_data(sess, args.sample)
    print ("Done...\n")
    depth = 581
    batch_size = 100

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 100], dtype=tf.float32)
    y_target = tf.placeholder(shape=[depth, None], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, 100], dtype=tf.float32)
    
    # Initialize gamma
    gamma = tf.constant(-5.0)
    
    # Create variables for svm
    b = tf.Variable(tf.random_normal(shape=[depth, batch_size]))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
    prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

    # Initialize variables
    save_path = "../trained_models/svm.ckpt"
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    # test_model
    sess.run(init)
    saver.restore(sess, save_path)
    print ("Model restored from file: %s" %save_path)

    
    # Testing loop
    i = 0
    test_batch_accuracy = []
    while (i + 1) * batch_size < len(x_test):
        
        index = [i for i in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = x_test[index]
        np_y = y_test_sparse[index].toarray()
        rand_y = np_y.transepose()
        
        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid:rand_x})
        test_batch_accuracy.append(acc_temp)
        print('Batch accuracy: ' + str(acc_temp))
        print('\n')
        print('\n')
        i += 1
    print('total accuracy : ' + str(sum(test_batch_accuracy) / float(len(test_batch_accuracy))))
    #save_model_metrics("rbf_model",  used_test_y, predicted, label_encoder)

if __name__=='__main__':
    main()

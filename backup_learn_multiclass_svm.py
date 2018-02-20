import argparse
import sys
import os
import gzip
from collections import Counter

import numpy as np
from Bio import SeqIO

from tensorflow.contrib import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.contrib.learn.python import SKCompat

from sklearn import svm
import tensorflow.contrib.learn as skflow
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, coverage_error
from scipy.spatial.distance import cosine
import cPickle as pickle
#from sklearn_theano.model_selection import cross_val_score


def get_sample(file_path):
    families = []
    vectors = []
    with open(file_path) as infile:
        for line in infile:
            uniprot_id, family, vector_string = line.rstrip().split('\t', 2)
            families.append(family)
            vectors.append(np.array(map(float, vector_string.split()), dtype=np.float32))  

    vectors_array = np.array(vectors)
    vectors = None

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(families)
    number_of_classes = len(set(label_encoder.classes_))

    families_encoded = np.array(label_encoder.transform(families), dtype=np.int32)
    families = None

    vectors_train, vectors_test, families_train, families_test = train_test_split(vectors_array, families_encoded, test_size=0.2, random_state=42)

    vectors_array, families_encoded, families_binary_labels = None, None, None

    min_on_training = vectors_train.min(axis=0)

    range_on_training = (vectors_train - min_on_training).max(axis=0)

    vectors_train_scaled = (vectors_train - min_on_training) / range_on_training
    vectors_test_scaled = (vectors_test - min_on_training) / range_on_training

    return label_encoder, vectors_train_scaled, vectors_test_scaled, families_train, families_test


def save_model_metrics(model_params_string, vectors_test, families_test, predicted_families, label_encoder):
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        #outfile.write('score: {}\n'.format(model.score(vectors_test, families_test)))
        #print('cross_val_test', cross_val_score(model, vectors_test, families_test, scoring='neg_log_loss'))
        #print('cross_val_train', cross_val_score(model, vectors_train, families_train, scoring='neg_log_loss'))
        #outfile.write('f1_macro: {}\n'.format(f1_score(families_test, predicted_families, average='macro')))
        #outfile.write('f1_micro: {}\n'.format(f1_score(families_test, predicted_families, average='micro')))
        #outfile.write('f1_weighted: {}\n'.format(f1_score(families_test, predicted_families, average='weighted')))
        #outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))

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
    parser.add_argument('--sample', type=str, default='training_sample_100.txt')
    parser.add_argument('--type', type=str, default='svc_linear')
    args = parser.parse_args()

    label_encoder, vectors_train, vectors_test, families_train, families_test = get_sample(args.sample)
    """
    if args.type == 'svc_linear':
        model = svm.SVC(kernel='linear', C=1000, gamma=10)
    elif args.type == 'svc_rbf':
        model = svm.SVC(kernel='rbf', C=10, gamma=0.1) 
    elif args.type == 'linear_svc':
        model = svm.LinearSVC() 
    """
    #feats = skflow.infer_real_valued_columns_from_input(vectors_train)
    real_feature_column = real_valued_columns("actual_family")
    sparse_feature_column = sparse_column_with_hash_bucket("predicted_family")

    #classifier_tf = SKCompat(skflow.DNNClassifier(feature_columns=feats, hidden_units=[50,50,50], n_classes=3, model_dir='./'))

    est = SKCompat(estimator)
    #estimator = SVM(example_id_column='example_id', feature_columns=[real_feature_column, sparse_feature_column], 12_reqularization=10.0)

    classifier_tf.fit(vectors_train, families_train, steps=5000)
    #est.fit(vectors_train

    predicted_families = list(classifier_tf.predict(vectors_test, as_iterable=True))

    score = metrics.accuracy_score(families_test, predicted_families)
    print("Accuracy: %f" % score)

    model_params_string = '{}_{}'.format(args.type, os.path.basename(args.sample))
    #with open('{}.pkl'.format(model_params_string), 'wb') as outfile:
    #    pickle.dump(model, outfile)

    save_model_metrics(model_params_string, vectors_test, families_test, predicted_families, label_encoder)

    #with open('svm_model.pkl', 'rb') as infile:
    #    model = pickle.load(infile)


if __name__ == '__main__':
    main()

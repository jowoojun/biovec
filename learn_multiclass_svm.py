import argparse
import sys
import os
import gzip
from collections import Counter
import cPickle as pickle

import numpy as np
from scipy.spatial.distance import cosine
from Bio import SeqIO

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, coverage_error
from sklearn.model_selection import cross_val_score


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

    vectors_train, vectors_test, families_train, families_test = train_test_split(vectors_array, families_encoded, test_size=0.12,random_state=42)
    vectors_array, families_encoded, families_binary_labels = None, None, None

    min_on_training = vectors_train.min(axis=0)

    range_on_training = (vectors_train - min_on_training).max(axis=0)

    vectors_train_scaled = (vectors_train - min_on_training) / range_on_training
    vectors_test_scaled = (vectors_test - min_on_training) / range_on_training

    return label_encoder, vectors_train_scaled, vectors_test_scaled, families_train, families_test


def save_model_metrics(model_params_string, model, vectors_test, families_test, label_encoder):
    with open('{}_results.txt'.format(model_params_string), 'w') as outfile:
        predicted_families = model.predict(vectors_test)
        outfile.write('score: {}\n'.format(model.score(vectors_test, families_test)))
        #print('cross_val_test', cross_val_score(model, vectors_test, families_test, scoring='neg_log_loss'))
        #print('cross_val_train', cross_val_score(model, vectors_train, families_train, scoring='neg_log_loss'))
        outfile.write('f1_macro: {}\n'.format(f1_score(families_test, predicted_families, average='macro')))
        outfile.write('f1_micro: {}\n'.format(f1_score(families_test, predicted_families, average='micro')))
        outfile.write('f1_weighted: {}\n'.format(f1_score(families_test, predicted_families, average='weighted')))
        outfile.write('accuracy_score: {}\n'.format(accuracy_score(families_test, predicted_families)))

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
    model = None
    if args.type == 'svc_linear':
        model = svm.SVC(kernel='linear', C=1000, gamma=10) 
    elif args.type == 'svc_rbf':
        model = svm.SVC(kernel='rbf') 
    elif args.type == 'linear_svc':
        model = svm.LinearSVC() 

    model.fit(vectors_train, families_train)


    model_params_string = '{}_{}'.format(args.type, os.path.basename(args.sample))
    with open('{}.pkl'.format(model_params_string), 'wb') as outfile:
        pickle.dump(model, outfile)

    save_model_metrics(model_params_string, model, vectors_test, families_test, label_encoder)
    print "trained_accurancy: {:.3f}".format(model.score(vectors_train, families_train))
    print "test_accrurancy  : {:.3f}".format(model.score(vectors_test, families_test)) 

    #with open('svm_model.pkl', 'rb') as infile:
    #    model = pickle.load(infile)


if __name__ == '__main__':
    main()

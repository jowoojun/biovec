import numpy
import pandas
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, coverage_error
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
"""
# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
print X
Y = dataset[:,4]
print Y

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
"""    
def get_sample(file_path):
    families = []
    vectors = []
    with open(file_path) as infile:
        for line in infile:
            uniprot_id, family, vector_string = line.rstrip().split('\t', 2)
            families.append(family)
            vectors.append(numpy.array(map(float, vector_string.split()), dtype=numpy.float32))  

    vectors_array = numpy.array(vectors)
    vectors = None

    encoder = LabelEncoder()
    encoder.fit(families)
    encoded_Y = encoder.transform(families)
    
    encoder = None

    vectors_train, vectors_test, families_train, families_test = train_test_split(vectors_array, encoded_Y, random_state = 0)

    return vectors_train, families_train, vectors_test, families_test, encoded_Y


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=100, activation='relu'))
	model.add(Dense(5592, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

X, Y, X_test, Y_test, encoded_Y = get_sample("trained_models/protein_pfam_vector.csv")


# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)
results = cross_val_score(estimator, X_test, Y_test, cv=kfold)
#results = accuracy_score(Y, dummy_y)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



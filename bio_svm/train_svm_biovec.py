import argparse

import numpy as np

import tensorflow as tf
import pandas
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.sparse import csc_matrix
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
parser.add_argument('--sample', type=str, default='../trained_models/protein_pfam_vector.csv')
args = parser.parse_args()

sess = tf.Session()

print ("Start getting data...")
#label_encoder, x_vals, y_vals, depth, data_size =g et_data(sess, args.sample)
print("Read_csv...")

dataframe = pandas.read_csv(args.sample, header=None)
dataset = dataframe.values
family = dataset[:,1]
vectors = dataset[:,2:].astype(float)
data_size = len(family)

print("Done...\n")

print("Labeling...")
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(family)
families_encoded = np.array(label_encoder.transform(family), dtype=np.int32)
family = None
depth = families_encoded.max() + 1
print("Done...\n")

famous_list = []
family_count = {}
for family in families_encoded:
    if family in family_count:
        family_count[family] += 1
    else:
        family_count[family] = 1
        
    if family_count[family] >= 500 and family not in famous_list:
        famous_list.append(family)

print("One hot Encoding...")
rows = np.arange(families_encoded.size)
cols = families_encoded
data = np.ones(families_encoded.size)
y_vals = csc_matrix((data, (rows, cols)), shape=(families_encoded.size, depth))

print("Done...\n")

min_on_training = vectors.min(axis=0)
range_on_training = (vectors - min_on_training).max(axis=0)

x_vals = (vectors - min_on_training) / range_on_training

print ("Done...\n")

batch_size = 250
learning_rate = 0.01

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 100], dtype=tf.float32)
y_target = tf.placeholder(shape=[depth, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 100], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[depth, batch_size]), name="b")

# Gaussian (RBF) kernel
gamma = tf.constant(-100.0)
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
prediction_onehot = tf.transpose(tf.one_hot(prediction, depth))

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

# Calculate confusion_matrix
#confusion_matrix = tf.confusion_matrix(y_target, prediction_onehot)

# Count true positives, true negatives, false positives and false negatives.
tp = tf.count_nonzero(prediction_onehot * y_target)
tn = tf.count_nonzero((prediction_onehot - 1) * (y_target - 1))
fp = tf.count_nonzero(prediction_onehot * (y_target - 1))
fn = tf.count_nonzero((prediction_onehot - 1) * y_target)

# Calculate accuracy, precision, recall and F1 score.
accuracy_with_confusion = tf.divide((tp + tn) , (tp + fp + fn + tn))
precision = tf.divide(tp , (tp + fp))
recall = tf.divide(tp , (tp + fn))
fmeasure = tf.divide((2 * precision * recall) , (precision + recall))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
init_op = tf.initialize_all_variables()

sess.run(init)
sess.run(init_op)


# loss and accuracy array declaration
loss_vec = []
test_batch_accuracy = []


#Initialize KFOLD Object
seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


#K fold cross validation
for train_index, test_index in kfold.split(x_vals, y_vals.toarray()):
    train_set, test_set = x_vals[train_index], x_vals[test_index]
    sparse_encoded_train_label, sparse_encoded_test_label = y_vals[train_index], y_vals[test_index]
    i = 0

    while (i + 1) * batch_size < len(train_set):
        index = [j for j in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = train_set[index]
        np_y = sparse_encoded_train_label[index].toarray()
        rand_y = np_y.transpose()
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        i += 1
        
        if (i+1)%25==0:
            print('train_Step #' + str(i+1))
            print('Loss = ' + str(temp_loss))
            
    i = 0
    while (i + 1) * batch_size < len(test_set):
        index = [j for j in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = test_set[index]
        np_y = sparse_encoded_test_label[index].toarray()
        rand_y = np_y.transpose()
        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})

        accuracy_with_confusion_val = sess.run(accuracy_with_confusion, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
        
        
        test_batch_accuracy.append(acc_temp)
        
        if (i+1)%25==0:
            print('\ntest_Step #' + str(i+1))
            print(',test_accuracy = ' + str(acc_temp))
            print('\nconfusion_accuracy = ' + str(accuracy_with_confusion_val))
            
        i += 1

    print('Batch accuracy: ' + str(acc_temp))
    print('\n')
    print('\n')
print(test_batch_accuracy)
print('Total accuracy: ' + str(float(sum(test_batch_accuracy)) / float(len(test_batch_accuracy))))

# Test with famous families 
for famous_family_num in famous_list:
    print('=======family_name = {}======'.format(label_encoder.inverse_transform(famous_family_num)))
    indices = []
    counter = 0
    for family_num in families_encoded:
        if famous_family_num == family_num:
            indices.append(counter)
        counter += 1
    i = 0
    batch_accuracy = []
    while (i + 1) * batch_size < len(indices):
        index = indices[i * batch_size : (i+1) * batch_size]
        rand_x = vectors[index]
        np_y = y_vals[index].toarray()
        rand_y = np_y.transpose()
        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    
        accuracy_with_confusion_val = sess.run(accuracy_with_confusion, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
        
        
        batch_accuracy.append(acc_temp)

        print('\ntest_Step #' + str(i+1))
        print(',test_accuracy = ' + str(acc_temp))
        print('\nconfusion_accuracy = ' + str(accuracy_with_confusion_val))
        print('Total accuracy: ' + str(float(sum(batch_accuracy)) / float(len(batch_accuracy))))
        i += 1



# =============================================================================
# with open('rbf_test_model_results.txt', 'w') as outfile:
#     outfile.write('accuracy_score: {}\n'.format(metrics.accuracy_score(families_test, predicted_families)))
#     
#     outfile.write('{}\t{}\t\n'.format(actual_family, accuracy))
#         
#     print(total_acc.eval(session=sess), update_op.eval(session=sess))
#     tp_rate = float(prediction_counter[True]) / sum(prediction_counter.values())
#     outfile.write('counter = {} TP_rate = {}\n'.format(prediction_counter, tp_rate))
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import pandas as df

from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Importing the dataset
dataframe = df.read_csv("./trained_models/protein_pfam_vector.csv", header=None)
dataset = dataframe.values
family = dataset[:, 1]
vectors = dataset[:,2:].astype(float)


seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Family = LabelEncoder()
family_label_encoding = labelencoder_Family.fit_transform(family)
depth = family_label_encoding.max()+1
    

#one hot encoded and make sparse matrix
from scipy.sparse import csc_matrix
rows = np.arange(family_label_encoding.size)
cols = family_label_encoding
data = np.ones(family_label_encoding.size)
sparse_one_hot = csc_matrix((data, (rows, cols)), shape=(family_label_encoding.size, family_label_encoding.max()+1))


batch_size = 250

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 100], dtype=tf.float32)
y_target = tf.placeholder(shape=[depth, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 100], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[depth, batch_size]))

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
prediction = tf.argmax(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_batch_accuracy = []

#K fold cross validation
for train_index, test_index in kfold.split(vectors, sparse_one_hot.toarray()):
    
    train_set, test_set = vectors[train_index], vectors[test_index]
    encoded_train_label, encoded_test_label = sparse_one_hot[train_index].toarray(), sparse_one_hot[test_index].toarray()
    i = 0
    while (i + 1) * batch_size < len(train_set):
        index = [i for i in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = train_set[index]
        rand_y = encoded_train_label[index].transpose()
        
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        i += 1
        
        if (i+1)%25==0:
            print('train_Step #' + str(i+1))
            print('Loss = ' + str(temp_loss))
            
    i = 0
    while (i + 1) * batch_size < len(test_set):
        index = [i for i in range(batch_size * i, batch_size * (i + 1) )]
        rand_x = test_set[index]
        rand_y = encoded_test_label[index].transpose()
        
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)   
        
        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid:rand_x})
        test_batch_accuracy.append(acc_temp)
        
        if (i+1)%25==0:
            print('test_Step #' + str(i+1))
            print(',Loss = ' + str(temp_loss))
            print(',test_accuracy = ' + str(acc_temp)) 
            print('\n')
            
        i += 1

    print('Batch accuracy: ' + str(acc_temp))
    print('\n')
    print('\n')

print('Total accuracy: ' + str(sum(test_batch_accuracy) / float(len(test_batch_accuracy))))
print('\n')
print('\n')
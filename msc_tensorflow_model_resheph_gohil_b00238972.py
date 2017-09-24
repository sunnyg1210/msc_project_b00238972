import matplotlib.pyplot as pyplot
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split as tts
from tensorflow.examples.tutorials.mnist import input_data

df = pd.read_csv('data/mldata_tensorflow.csv')
X_init = df.drop(['time_in_hospital', 'los'], axis=1).values
y_init = df.los

# sampling the data within X and y into four seperate variables
X_train, X_test, y_train, y_test = tts(X_init, y_init, random_state=40, train_size=0.7, test_size=0.3)
x_train_len = len(X_train)

def one_hot_encode(labels):
    nb_label_classes = 3
    one_hot_targets = np.eye(nb_label_classes)[labels]
    return one_hot_targets

print('Performing one hot encode...')
encoded_y_train = one_hot_encode(y_train)
encoded_y_test = one_hot_encode(y_test)

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

n_classes = 3
batch_size = 100

x = tf.placeholder('float', [None, 16])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([16, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output

learning_rate = 0.3
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.3).minimize(cost)

    hm_epochs = 10
    with tf.Session(config = config) as sess:
        
        sess.run(tf.global_variables_initializer())
        print('\nTensorFlow Session Started...\n')
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < x_train_len:
                start = i
                end = i + batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(encoded_y_train[start:end])
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of ', hm_epochs, '; loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: X_test, y: encoded_y_test}))

train_neural_network(x)
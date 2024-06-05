''' benchmark_mnist.py '''
import os
import sys
import time
import argparse as argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.client import timeline 
from sklearn.datasets import fetch_openml


# import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def benchmark_mnist(args): 
    # load data
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) 
    
    
    #mnist = input_data.read_data_sets(args.data_dir, one_hot=True) 
    # create model
    x = tf.placeholder(tf.float32, [None, 784]) 
    y = tf.placeholder(tf.float32, [None, 10])
    logits = model.inference(x)
    loss = model.loss(logits, y)
    train_op = model.train(loss)
    # evaluation
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
    # initialize variables
    init = tf.global_variables_initializer()
    # create session
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    start = time.time() 

    sess.run(init) 
    # create summary writer
    if not args.no_log:
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph) 
    # run training
    for epoch in range(args.num_epochs):
        for batch in range(args.num_batches):
            batch_x, batch_y = mnist.train.next_batch(args.batch_size)
            feed_dict = {x: batch_x, y: batch_y}
            if not args.no_log:
                _, summary = sess.run([train_op, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, epoch * args.num_batches + batch)
            else:
                sess.run(train_op, feed_dict=feed_dict) 
    # run evaluation
    feed_dict = {x: mnist.test.images, y: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict)
    end = time.time()
    # print results
    print('Accuracy: %f' % acc)
    print('Time: %f' % (end - start))
    # close session
    sess.close()
    
    # confusion matrix
    y_true = np.argmax(mnist.test.labels, axis=1)
    y_pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # accuracy
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    # precision
    precision = precision_score(y_true, y_pred, average='macro')
    print(precision)
    # recall
    recall = recall_score(y_true, y_pred, average='macro')
    print(recall)
    # f1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f1)
    # plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('mnist_tf_confusion_matrix.png')


def parse_args():
    ''' parse arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    return parser.parse_args()

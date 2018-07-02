# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:32:45 2018

@author: yu
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib.pylab as plt
import pickle
import glob
import EarthQuakePlateModelKDE_FFT as eqp

#---------------------
# parameter setting
#nCell = 8
nCell = 1
nClass = 13
nWindow = 10
bInd = 0
size = 25
#---------------------

#---------------------
# Define some handy network layers
def weight_variable(name,shape):
	return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
	return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))

def softmax(inputs,w,b,keepProb):
	softmax = tf.matmul(inputs,w) + b
	softmax = tf.nn.dropout(softmax, keepProb)
	softmax = tf.nn.softmax(softmax)
	return softmax

def fc_relu(inputs, w, b, keepProb):
	relu = tf.matmul(inputs, w) + b
	relu = tf.nn.dropout(relu, keepProb)
	relu = tf.nn.relu(relu)
	return relu

def fc(inputs, w, b, keepProb):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	return fc

#---------------------

#---------------------
# regression
def nn(x,reuse=False,nCell=1,nWindow=10, nHidden = 10):
	with tf.variable_scope('nn') as scope:  
		keepProb = 1
		if reuse:
			keepProb = 1.0			
			scope.reuse_variables()

		#input -> hidden
		w1 = weight_variable('w1',[nCell*nWindow,nHidden])
		b1 = bias_variable('b1',[nHidden])
		h = fc_relu(x,w1,b1,keepProb) 

		#hidden -> output
		w2 = weight_variable('w2',[nHidden,1])
		b2 = bias_variable('b2',[1])
		y = fc_relu(h,w2,b2,keepProb)

		return y
#---------------------

#---------------------
# classification
def nn_class(x,reuse=False,nCell=1,nWindow=10, nHidden = 10):
	with tf.variable_scope('nn_class') as scope:  
		keepProb = 1
		if reuse:
			keepProb = 1.0			
			scope.reuse_variables()

		#input -> hidden
		w1 = weight_variable('w1',[nCell*nWindow,nHidden])
		b1 = bias_variable('b1',[nHidden])
		h = fc_relu(x,w1,b1,keepProb) 

		#hidden -> output
		w2 = weight_variable('w2',[nHidden,nClass])
		b2 = bias_variable('b2',[nClass])
		y = fc(h,w2,b2,keepProb)

		return y
#---------------------

# load data
myData = eqp.Data(fname="kde_fft_log_10*", trainRatio=0.8, bInd=bInd, isWindows=False, isClass=True)

# Build the computation graph for training
x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow])

# regression
y = tf.placeholder(tf.float32, shape=[None])

# classification
y_class = tf.placeholder(tf.float32, shape=[None,nClass])

# regression
predict_op = nn(x)
predict_test_op = nn(x,reuse=True)

# classification
predict_class_op = nn_class(x)
predict_class_test_op = nn_class(x,reuse=True)
#---------------------

#---------------------
# loss
loss = tf.reduce_mean(tf.abs(predict_op - y))
loss_test = tf.reduce_mean(tf.abs(predict_test_op - y))
loss_class = tf.losses.softmax_cross_entropy(y_class, predict_class_op)
loss_test_class = tf.losses.softmax_cross_entropy(y_class, predict_class_test_op)

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
trainer_class = tf.train.AdamOptimizer(1e-3).minimize(loss_class)
#---------------------

#---------------------
# accuracy
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_class, 1), tf.argmax(predict_class_op, 1)), tf.float32))

#---------------------
batchSize = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())

testX = np.reshape(myData.xTest[:,0,:],[-1,nCell*nWindow])
testY = myData.yTest 

# Start training
for i in range(100000):
	
	batchX, batchY = myData.nextBatch(batchSize)
	#batchX = np.reshape(batchX,[-1,nCell*nWindow])
	# only 1st cell
	batchX = np.reshape(batchX[:,0,:],[-1,nCell*nWindow])

	# regression	
	#_, lossTrain, predTrain = sess.run([trainer, loss, predict_op], feed_dict={x:batchX, y:batchY})

	# classification
	_, lossTrain, predTrain = sess.run([trainer_class, loss_class, predict_class_op], feed_dict={x:batchX, y_class:batchY})
	
	if i % 100 == 0:
		print("iteration: %d,loss: %f" % (i,lossTrain))
	if i % 500 == 0:
		# regression
		#lossTest, predTest  = sess.run([loss_test, predict_test_op], feed_dict={x:batchX, y:batchY})

		# classification
		#lossTest, predTest, accuracy  = sess.run([loss_test_class, predict_class_test_op, accuracy_op], feed_dict={x:batchX, y_class:batchY})
		lossTest, predTest, accuracy  = sess.run([loss_test_class, predict_class_test_op, accuracy_op], feed_dict={x:testX, y_class:testY})
		print("iteration: %d,loss: %f, accuracy: %f" % (i,lossTest,accuracy))

		print("--------------------------")
		print("testX:\n",testX[:10])
		print("--------------------------")
		print("predTest:\n",predTest[:10])
		print("argmax predTest:",np.argmax(predTest[:10],axis=1))
		print("argmax testY:",np.argmax(testY[:10],axis=1))
		print("--------------------------")
		
		with open("training/process_{}.pickle".format(i), "wb") as fp:
			pickle.dump(predTest,fp)
			pickle.dump(myData.xTest,fp)
			pickle.dump(myData.yTest,fp)
			pickle.dump(lossTest,fp)

		# save model to file
		saver = tf.train.Saver()
		saver.save(sess,"models/model_{}.ckpt".format(i))
#---------------------


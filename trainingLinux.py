# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib.pylab as plt
import pickle
import pywt
import glob
import EarthQuakePlateModelLinux as eqp

#----------------
# Define some handy network layers
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
	
def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
	
def conv2d_relu(inputs, w, b, strides):
	#filter: [kernel, output_depth, input_depth]
	conv = tf.nn.conv2d(inputs, w, strides=strides, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv	

def pool2d(inputs, ksize, strides):
	pool = tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding='SAME')
	return pool

def fc_relu(inputs, w, b, keepProb):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	fc = tf.nn.relu(fc)
	return fc
#----------------

def cnn(x, reuse=False):
	with tf.variable_scope('cnn') as scope:
		keepProb = 0.5
		if reuse:
			keepProb = 1.0
			scope.reuse_variables()
	
		# when padding='SAME', O = I/S
		# 1000/2 = 500, 500/2 = 250
		convW1 = weight_variable("convW1", [3, 5, 8, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_relu(x, convW1, convB1, strides=[1,2,2,1])
		conv1 = pool2d(conv1,ksize=[1,1,2,1], strides=[1,1,2,1])
		
		# 250/2 = 125, 125/2 = 63
		convW2 = weight_variable("convW2", [3, 5, 32, 64])
		convB2 = bias_variable("convB2", [64])
		conv2 = conv2d_relu(conv1, convW2, convB2, strides=[1,2,2,1])
		conv2 = pool2d(conv2,ksize=[1,1,2,1], strides=[1,1,2,1])
				
		# 63/2 = 32, 32/2 = 16
		convW3 = weight_variable("convW3", [3, 5, 64, 64])
		convB3 = bias_variable("convB3", [64])
		conv3 = conv2d_relu(conv2, convW3, convB3, strides=[1,2,2,1])
		conv3 = pool2d(conv3,ksize=[1,1,2,1], strides=[1,1,2,1])
		
		# 16/2 = 8
		convW4 = weight_variable("convW4", [3, 5, 64, 64])
		convB4 = bias_variable("convB4", [64])
		conv4 = conv2d_relu(conv3, convW4, convB4, strides=[1,1,2,1])
		
		# convert to vector
		conv4_dim = np.prod(conv4.get_shape().as_list()[1:])
		conv4 = tf.reshape(conv4, [-1, conv4_dim])
		
		# 9*9*32 = 2592 -> 128
		fcW1 = weight_variable("fcW1", [conv4_dim, 128])
		fcB1 = bias_variable("fcB1", [128])
		fc1 = fc_relu(conv4, fcW1, fcB1, keepProb)
		
		# 128 -> 20
		fcW2 = weight_variable("fcW2", [128, 1])
		fcB2 = bias_variable("fcB2", [1])
		fc2 = fc_relu(fc1, fcW2, fcB2, keepProb)
		return fc2
		
#---------------------
# load data
nCell = 8
nFreqs = 30
sYear = 1000
eYear = 2000
bInd = 0
myData = eqp.Data(fname="log_10*",trainRatio=0.8,nCell=nCell,nFreqs=nFreqs, sYear=sYear, eYear=eYear, bInd=bInd, isTensorflow=True)
#---------------------

#---------------------
# Build the computation graph for training
xTrain = tf.placeholder(tf.float32, shape=[None, nFreqs, eYear-sYear, nCell])
yTrain = tf.placeholder(tf.float32, shape=[None])
predict_op = cnn(xTrain)
#---------------------

#---------------------
# loss
loss_op = tf.reduce_mean(tf.square(predict_op - yTrain))
trainer = tf.train.AdamOptimizer(1e-3).minimize(loss_op)
#---------------------

#---------------------
batchSize = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start training
for i in range(10500):
	batchX, batchY = myData.nextBatch(batchSize)
	
	_, lossTrain = sess.run([trainer, loss_op], feed_dict={xTrain:batchX, yTrain:batchY})
	
	if i % 100 == 0:
		print("iteration: %d, loss: %f" % (i,lossTrain))
	if i>0 & i % 500 == 0:
		lossTest, predictTest  = sess.run([loss_op,predict_op], feed_dict={xTrain:myData.xTest, yTrain:myData.yTest})

		with open("training/process_{}.pickle".format(i), "wb") as fp:
			pickle.dump(predictTest,fp)
			pickle.dump(lossTest,fp)

		# save model to file
		saver = tf.train.Saver()
		saver.save(sess,"../models/model_{}.ckpt".format(i))
#---------------------


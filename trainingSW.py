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
import EarthQuakePlateModel_KDE as eqp

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

def relu(inputs, w, b, keepProb):
	relu = tf.matmul(inputs, w) + b
	relu = tf.nn.dropout(relu, keepProb)
	relu = tf.nn.relu(relu)
	return relu


#---------------------

#---------------------

def nn(x,reuse=False):
    with tf.variable_scope('nn') as scope:  
        #keepProb = 0.5
        if reuse:
            keepProb = 1.0            
            scope.reuse_variables()
        
        # Tensorflow用にデータを整形
        x = tf.transpose(x,(1,0,2))
    
        #input層
        w1 = weight_variable('w1',[8,31,1])
        b1 = bias_variable('b1',[1])
        y1 = relu(x,w1,b1,0.2)# [8,None,1]
        
        
        y1 = tf.transpose(y1,(1,0,2)) # [None,8,1]
        # 3d -> 2d
        y1 = tf.reshape(y1,tf.stack([-1,np.prod(y1.get_shape().as_list()[1:])])) # [None,8]
        
        #hidden_1層
        w2 = weight_variable('w2',[np.prod(y1.get_shape().as_list()[1:]),4])
        b2 = bias_variable('b2',[4])
        y2 = relu(y1,w2,b2,0.5)
        
        #hidden_2層
        #w3 = weight_variable('w3',[np.prod(y2.get_shape().as_list()[1:]),2])
        #b3 = bias_variable('b3',[2])
        #y3 = relu(y2,w3,b3,keepProb)
        
        #output層
        w4 = weight_variable('w4',[np.prod(y2.get_shape().as_list()[1:]),1])
        b4 = bias_variable('b4',[1])
        y4 = relu(y2,w4,b4,1.0)
        
        # 2d -> 1d
        y4 = tf.reshape(y4,tf.stack([-1,]))
        return y4            
            
            
# load data
nCell = 8
nSlice = 31
sYear = 2000
eYear = 10000
bInd = 0
size = 25

myData = eqp.Data(fname="yVlog_10*",trainRatio=0.8,nCell=nCell,sYear=sYear, eYear=eYear, bInd=bInd, isTensorflow=False)
kde = myData.KDE(nYear=8000)
features = myData.FFT(slice_size=size,eFrq=250,nYear=8000,nCell=nCell,trainRatio=0.8,isTensorflow=True)

# Build the computation graph for training
x = tf.placeholder(tf.float32, shape=[None,nCell,nSlice])
y_ = tf.placeholder(tf.float32, shape=[None])

predict_op = nn(x)
predict_test_op = nn(x,reuse=True)
#---------------------

#---------------------
# loss
loss= tf.reduce_mean(tf.abs(predict_op - y_))
loss_test= tf.reduce_mean(tf.abs(predict_test_op - y_))

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
#---------------------

#---------------------
batchSize = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Start training
for i in range(70000):
    
    batchX, batchY = myData.nextBatch(batchSize)
    
    
    _, lossTrain, predTrain = sess.run([trainer, loss, predict_op], feed_dict={x:batchX, y_:batchY})
    
    
    if i % 1000 == 0:
        print("iteration: %d,loss: %f" % (i,lossTrain))
        print(batchY)
        print("Train b:",predTrain)
    if i % 500 == 0:
        lossTest, predTest  = sess.run([loss_test, predict_test_op], feed_dict={x:features[2], y_:features[3]})
        
        with open("training/process_{}.pickle".format(i), "wb") as fp:
            pickle.dump(predTest,fp)
            pickle.dump(myData.xTest,fp)
            pickle.dump(myData.yTest,fp)
            pickle.dump(lossTest,fp)
            #pickle.dump(myData.minY,fp)
            #pickle.dump(myData.maxY,fp)

        # save model to file
        saver = tf.train.Saver()
        saver.save(sess,"models/model_{}.ckpt".format(i))
#---------------------


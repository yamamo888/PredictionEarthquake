import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import math, os, sys
import pickle
import pdb
import matplotlib.pylab as plt


class Training():
    
    
    def __init__(self,zdims=80):
        # 最終的に出力する次元数
        self.zdims = zdims    
        self.cell = 1
        
    
    
    def weight_variable(self,name, shape):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
        
    def bias_variable(self,name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
        
    def conv1d_relu(self,inputs, w, b, stride):
        #filter: [kernel, output_depth, input_depth]
        conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
        conv = tf.nn.relu(conv)
        return conv
    
    def conv1d_t_relu(self,inputs, w, b, output_shape, stride):
        conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
        conv = tf.nn.relu(conv)
        return conv
        
    def fc_relu(self,inputs, w, b):
        fc = tf.matmul(inputs, w) + b
        fc = tf.nn.relu(fc)
        return fc
    
    def cnn(self,x,z_dim,reuse=False):
        with tf.variable_scope('cnn') as scope:
            if reuse:
                scope.reuse_variables()
        
            # 8000 -> 800
            # 1layer
            convW1 = self.weight_variable("convW1", [10, 1, 32])
            convB1 = self.bias_variable("convB1", [32])
            conv1 = self.conv1d_relu(x, convW1, convB1, stride=10)
            
            # 800 -> 80
            # 2layer
            convW2 = self.weight_variable("convW2", [8, 32, 64])
            convB2 = self.bias_variable("convB2", [64])
            conv2 = self.conv1d_relu(conv1, convW2, convB2, stride=10)
            
            # 80 -> 8
            # 3layer
            convW3 = self.weight_variable("convW3", [6, 64, 64])
            convB3 = self.bias_variable("convB3", [64])
            conv3 = self.conv1d_relu(conv2, convW3, convB3, stride=10)
            
            
            # convert to vector
            conv3 = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
            
            # 8x64 =  -> 128
            fcW1 = self.weight_variable("fcW1", [8*64, 128])
            fcB1 = self.bias_variable("fcB1", [128])
            fc1 = self.fc_relu(conv3, fcW1, fcB1)
            
            
            # 128 -> 80
            fcW2 = self.weight_variable("fcW2", [128, z_dim])
            fcB2 = self.bias_variable("fcB2", [z_dim])
            fc2 = self.fc_relu(fc1, fcW2, fcB2)
            
            return fc2
        
            

    
if __name__ == "__main__":
    
    
    
    mytraining = Training(zdims=80)
    zdims = mytraining.zdims
    # 今はテキトーなディレクトリ指定
    with open("visualization/yV.pickle",'rb') as fp:
        yV = pickle.load(fp)
    
    # ここで8000にしているが、本番はpickle化する
    # 入力
    yV1 = yV[:,0,2000:,np.newaxis]    
    yV2 = yV[:,1,2000:,np.newaxis]    
    yV3 = yV[:,2,2000:,np.newaxis]    
    yV4 = yV[:,3,2000:,np.newaxis]    
    yV5 = yV[:,4,2000:,np.newaxis]    
    yV6 = yV[:,5,2000:,np.newaxis]    
    yV7 = yV[:,6,2000:,np.newaxis]    
    yV8 = yV[:,7,2000:,np.newaxis]    
    
    #pdb.set_trace()
    # 出力    
    cnn_yV1 = mytraining.cnn(tf.cast(yV1,tf.float32),zdims)
    cnn_yV2 = mytraining.cnn(tf.cast(yV2,tf.float32),zdims)
    cnn_yV3 = mytraining.cnn(tf.cast(yV3,tf.float32),zdims)
    cnn_yV4 = mytraining.cnn(tf.cast(yV4,tf.float32),zdims)
    cnn_yV5 = mytraining.cnn(tf.cast(yV5,tf.float32),zdims)
    cnn_yV6 = mytraining.cnn(tf.cast(yV6,tf.float32),zdims)
    cnn_yV7 = mytraining.cnn(tf.cast(yV7,tf.float32),zdims)
    cnn_yV8 = mytraining.cnn(tf.cast(yV8,tf.float32),zdims)
    
    # 8cell
    cnn_yV = np.concatnate((cnn_yV1[:,np.newaxis,:],cnn_yV2[:,np.newaxis,:],
                   cnn_yV3[:,np.newaxis,:],cnn_yV4[:,np.newaxis,:],
                   cnn_yV5[:,np.newaxis,:],cnn_yV6[:,np.newaxis,:],
                   cnn_yV7[:,np.newaxis,:],cnn_yV8[:,np.newaxis,:],),1)
    
    
    

               
    
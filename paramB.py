# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:34:21 2018

@author: yu
"""
import os
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops

class Data:
    dataPath = '../CycleBranch2'
    
    
    def __init__(self,fname):
        fullpath = os.path.join(self.dataPath,fname)
        print('reading',fullpath)
    
        self.data = open(fullpath).readlines()
        
    #bとvのデータをnumpyに変更    
    def data_bv(self):
        
         Ncell = 8
         
         self.lista = []
         for i in np.arange(1,Ncell+1):
            datasplit = self.data[i].split(",")
            datasplit1 = datasplit[0]
            self.lista.append(datasplit1)
    
         self.listb = []
         for i in np.arange(1,Ncell+1):
            datasplit = self.data[i].split(",")
            datasplit1 = datasplit[1]
            self.listb.append(datasplit1)
        
         self.listL = []
         for i in np.arange(1,Ncell+1):
            datasplit = self.data[i].split(",")
            datasplit1 = datasplit[4]
            self.listL.append(datasplit1)
         
         # Vの開始行取得
         isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
         Vsind = np.where(isRTOL)[0][0]+1
         
         pdb.set_trace()
         # Vの値の取得（Vsind行から最終行まで）
         self.listv = []
         for j in np.arange(Vsind,len(self.data)):
            data = self.data[j].split(",")
            self.listv.append(data)
        
         pdb.set_trace()
         self.dataa = np.array(self.lista)
         self.datab = np.array(self.listb)
         self.dataL = np.array(self.listL)
         self.datav = np.array(self.listv)
         #pdb.set_trace()
         
    
        
#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    #Reading load log.txt
    
    #log1 = Data("log_1.txt").data_bv()
    #log2 = Data("log_2.txt").data_bv()
    #log3 = Data("log_3.txt").data_bv()
    #log4 = Data("log_4.txt").data_bv()
    #log5 = Data("log_5.txt").data_bv()
    #log6 = Data("log_6.txt").data_bv()
    #log7 = Data("log_7.txt").data_bv()
    #log8 = Data("log_8.txt").data_bv()
    #log9 = Data("log_9.txt").data_bv()
    log10 = Data("log_10.txt").data_bv()
    #pdb.set_trace()
    
################### tensor #######################
    z_dim = 10
    num_v = 16270 #1627*10
    batch_size = 200
    
    x = tf.placeholder(tf.float32,shape=[None,num_v,1])
    #y = tf.placeholder(tf.float32,shape=[None,,1])
    
    trainz = Data.encoder(x,z_dim)
    trainx = Data.decoder(trainz,z_dim)
    
    testz = Data.encoder(x,z_dim,reuse=True)
    testx = Data.decoder(testz,z_dim,reuse=True)
    
    true_samples = tf.random_normal(tf.stack[200,z_dim])
    loss_mmd = Data.compute_mmd(true_samples,trainz)
    loss_nll = tf.reduce_mean(tf.square(trainx))
    loss = loss_nll + loss_mmd
    trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    pdb.set_trace()
    
    for i in range(1000):
        batch_x = log1.nextBatch(batch_size)
        _, nll, mmd, train_xr, train_z = sess.run([trainer, loss_nll, loss_mmd, trainx, trainz], feed_dict={x: batch_x})
        #if i % 100 == 0:
         #   print("iteration: %d, Negative log likelihood is %f, mmd loss is %f" % (i,nll, mmd))
        #if i % 10000 == 0:
         #   test_xr,test_z = sess.run([test_xr_op,test_z_op], feed_dict={x: myLaser.xTest})
    
    # save model to file
        #saver = tf.train.Saver()
        #saver.save(sess,"../models/1d_{}.ckpt".format(i))"""

        
   
    
    """listdata = []
    for i in np.arange(1,9):
        datasplit = log1.file[i].split(",")
        datasplit1 = datasplit[1]
        listdata.append(datasplit1)
    
    for j in np.arange(83,1710):
        data = log1.file[j].split(",")
        listdata.append(data)
    
    print(listdata)
    pdb.set_trace()"""
    
    

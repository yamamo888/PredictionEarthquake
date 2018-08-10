# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:58:46 2018

@author: yu
"""
import os
import sys
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib.pylab as plt
import pickle
import glob
import EarthQuakePlateModelKDE_FFT as eqp

class Trainingeqp():
    
    def __init__(self,inputCellMode=1,dataMode=1,outputCellMode=1,datapickleMode=1):
        inputCellMode = int(sys.argv[1])
        dataMode = int(sys.argv[2])
        outputCellMode = int(sys.argv[3])
        datapickleMode = int(sys.argv[4])
        
        if inputCellMode == 1:
            nCell = 8
        elif inputCellMode == 2:
            nCell = 1
        elif inputCellMode == -1:
            nCell = 7
            #nCell = 2
        
        # b1単独予測器の場合は_1指定
        # b2単独予測器の場合は_2指定
        # b1b2組み合わせデータの場合は_12指定
        
        if dataMode == 12:
            dataPath = 'b1b2'
            picklePath='xydatab1b2.pkl'
            trainingpicklePath='traintestdatab1b2.pkl'
            fname='kde_fft_log_20*'
        
        elif dataMode == 123:
            dataPath = 'b1b2b3'    
            picklePath='xydatab1b2b3.pkl'
            trainingpicklePath='traintestdatab1b2b3.pkl'
            fname='kde_fft_log_20*'
        

        if outputCellMode == 1:
            bInd = 0    
        elif outputCellMode == 2:
            bInd = 1
        elif outputCellMode == 12:
            bInd=[0,1]
        elif outputCellMode == 23:
            bInd=[1,2]
        elif outputCellMode == 123:
            bInd=[1,2,3]
        
        # Mode
        self.inputCellMode = inputCellMode
        self.dataMode = dataMode
        self.outputCellMode = outputCellMode
        self.datapickleMode = datapickleMode
        
        # Path
        self.featuresPath = './features'
        self.modelsPath = './models'
        self.trainingPath = './training'

        self.dataPath = dataPath
        self.picklePath = picklePath
        self.trainingpicklePath = trainingpicklePath
        self.fname = fname

        
        
        
        # parameter setting
        self.nCell = nCell
        self.bInd = bInd
        self.nWindow = 10
        self.nHidden = 240
        self.multinHidden = 480
        self.nClass = 12
        self.size = 25
        
        self.myData = eqp.Data(fname=fname, trainRatio=0.8, nCell=nCell, sYear=2000, bInd=bInd, 
              eYear=10000, isWindows=True, isClass=True, 
              dataMode=dataMode, outputCellMode=outputCellMode, datapickleMode=datapickleMode,
              featuresPath=self.featuresPath,dataPath=dataPath,trainingpicklePath=trainingpicklePath,picklePath=picklePath)

        
    def nn_class(self,x,reuse=False):
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
        
        with tf.variable_scope('nn_class') as scope:  
            keepProb = 1
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            
            #input -> hidden
            self.w1 = weight_variable('w1',[self.nCell*self.nWindow,self.nHidden])
            self.b1 = bias_variable('b1',[self.nHidden])
            h = fc_relu(x,self.w1,self.b1,keepProb) 
    
            #hidden -> output
            if self.outputCellMode == 1 or self.outputCellMode == 2:
                self.w2 = weight_variable('w2',[self.nHidden,self.nClass])
                self.b2 = bias_variable('b2',[self.nClass])
                y = fc(h,self.w2,self.b2,keepProb)
                return y

    def TwoDimensions_nn_class(self,x,reuse=False):
        def weight_variable(name,shape,trainable):
            return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
            
        def bias_variable(name,shape,trainable):
            return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
            
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
        
        with tf.variable_scope('TwoDimensions_nn_class') as scope:  
            keepProb = 1
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
                
            #input -> hidden
            #学習させない時は、trainable=False(上書き値で固定)
            self.multiw1 = weight_variable('multiw1',[self.nCell*self.nWindow,self.multinHidden],trainable=False)
            self.b1 = bias_variable('b1',[self.multinHidden],trainable=False)
            h = fc_relu(x,self.multiw1,self.b1,keepProb) 
            
            #hidden -> output
            self.w2_1 = weight_variable('w2_1',[self.multinHidden,self.nClass],trainable=True)
            self.b2_1 = bias_variable('b2_1',[self.nClass],trainable=True)
            
            self.w2_2 = weight_variable('w2_2',[self.multinHidden,self.nClass],trainable=True)
            self.b2_2 = bias_variable('b2_2',[self.nClass],trainable=True)
            
            
            y1 = fc(h,self.w2_1,self.b2_1,keepProb)
            y2 = fc(h,self.w2_2,self.b2_2,keepProb)
            
            return y1, y2
            
    def ThreeDimensions_nn_class(self,x,reuse=False):
        def weight_variable(name,shape,trainable):
            return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
            
        def bias_variable(name,shape,trainable):
            return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
            
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
        
        with tf.variable_scope('ThreeDimensions_nn_class') as scope:  
            keepProb = 1
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
                
            #input -> hidden
            #学習させない時は、trainable=False(上書き値で固定)
            self.multiw1 = weight_variable('multiw1',[self.nCell*self.nWindow,self.multinHidden],trainable=False)
            self.b1 = bias_variable('b1',[self.multinHidden],trainable=False)
            h = fc_relu(x,self.multiw1,self.b1,keepProb) 
            
            #hidden -> output
            self.w2_1 = weight_variable('w2_1',[self.multinHidden,self.nClass],trainable=True)
            self.b2_1 = bias_variable('b2_1',[self.nClass],trainable=True)
            
            self.w2_2 = weight_variable('w2_2',[self.multinHidden,self.nClass],trainable=True)
            self.b2_2 = bias_variable('b2_2',[self.nClass],trainable=True)
            
            self.w2_3 = weight_variable('w2_3',[self.multinHidden,self.nClass],trainable=True)
            self.b2_3 = bias_variable('b2_3',[self.nClass],trainable=True)
            
            y1 = fc(h,self.w2_1,self.b2_1,keepProb)
            y2 = fc(h,self.w2_2,self.b2_2,keepProb) 
            y3 = fc(h,self.w2_3,self.b2_3,keepProb)
            return y1, y2
        
if __name__ == "__main__":
    
    isWindows = False
    
    # Mode
    inputCellMode = int(sys.argv[1])
    dataMode = int(sys.argv[2])
    outputCellMode = int(sys.argv[3])
    datapickleMode = int(sys.argv[4])
    
    mytraining = Trainingeqp(inputCellMode=inputCellMode,dataMode=dataMode,outputCellMode=outputCellMode,datapickleMode=datapickleMode)
    
    
    # parameter
    nCell = mytraining.nCell
    bInd = mytraining.bInd
    nWindow = mytraining.nWindow
    nClass = mytraining.nClass
    batchSize = 750
    
    # b1b2両方を出力したいときは True
    isTwoDimensionsPrediction = False
    
    if isTwoDimensionsPredicton:
        
        ######################### Multi Prediction ###########################################
        x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow])
        y1_class = tf.placeholder(tf.float32, shape=[None,nClass])
        y2_class = tf.placeholder(tf.float32, shape=[None,nClass])
        
        
        predict_class1_op,predict_class2_op = mytraining.Multinn_class(x)
        predict_class1_test_op,predict_class2_test_op = mytraining.Multinn_class(x,reuse=True)
        
        
        loss_class1 = tf.losses.softmax_cross_entropy(y1_class, predict_class1_op)
        loss_test_class1 = tf.losses.softmax_cross_entropy(y1_class, predict_class1_test_op)
    
        loss_class2 = tf.losses.softmax_cross_entropy(y2_class, predict_class2_op)
        loss_test_class2 = tf.losses.softmax_cross_entropy(y2_class, predict_class2_test_op)
    
        # loss of b1 + b2
        loss_class_all = loss_class1 + loss_class2
        loss_test_class_all = loss_test_class1 + loss_test_class2
    
        # training by classification
        trainer_class = tf.train.AdamOptimizer(1e-3).minimize(loss_class_all)
        #---------------------
        
        #---------------------
        # accuracy
        accuracy_op1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y1_class, 1), tf.argmax(predict_class1_op, 1)), tf.float32))
        accuracy_op2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y2_class, 1), tf.argmax(predict_class2_op, 1)), tf.float32))

        
        # training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #---------------------
        
        
        if inputCellMode == 1:
            testX = np.reshape(mytraining.myData.xTest,[-1,nCell*nWindow])
        elif inputCellMode == 2:
            testX = np.reshape(mytraining.myData.xTest[:,bInd,:],[-1,nCell*nWindow])
        elif inputCellMode == -1:
            # bIndのセルを削除して、bInd以外のセルを使う
            testX = np.delete(mytraining.myData.xTest,bInd,1)
            testX = np.reshape(testX,[-1,nCell*nWindow])
        
        # eqp から渡されたbが指定されている
        testY1 = mytraining.myData.y1Test
        testY2 = mytraining.myData.y2Test
        
        #process の番号を変更する
        if isWindows:
                w1PickleFiles = glob.glob('trainingw1\\process_*pickle')
                w2PickleFiles = glob.glob('trainingw2\\process_*pickle')
                
        else:
                w1PickleFiles = glob.glob('./training/b1b2/processb1_*pickle')
                w2PickleFiles = glob.glob('./training/b1b2/processb2_*pickle')
                
        
        # b1のw1train,test 
        with open(w1PickleFiles[0],"rb") as fp:
            w1_1Train = pickle.load(fp)
            w1_1Test = pickle.load(fp)
            b1_1Train = pickle.load(fp)
            b1_1Test = pickle.load(fp)
            

        # b2のw1train,test
        with open(w2PickleFiles[0],"rb") as fp:
            w1_2Train = pickle.load(fp)
            w1_2Test = pickle.load(fp)
            b1_2Train = pickle.load(fp)
            b1_2Test = pickle.load(fp)
            
        #b1b2のw1(b1)をconcateして、480次元のmultiw1にする
        w1Train = np.concatenate((w1_1Train,w1_2Train),axis=1)
        w1Test = np.concatenate((w1_1Test,w1_2Test),axis=1)
        
        b1Train = np.concatenate((b1_1Train,b1_2Train),axis=0)
        b1Test = np.concatenate((b1_1Test,b1_2Test),axis=0)
        
        
        # w1を上書きして、初期値変更
        w1TrainUpdate_op = tf.assign(mytraining.multiw1,w1Train)
        w1TestUpdate_op = tf.assign(mytraining.multiw1,w1Test)
        # b1を上書きして、初期値変更
        b1TrainUpdate_op = tf.assign(mytraining.b1,b1Train)
        b1TestUpdate_op = tf.assign(mytraining.b1,b1Test)                                             
        
        
        # Start training
        for i in range(100000):
            
            batchX, batchY1,batchY2 = mytraining.myData.nextBatch(batchSize)
            
            if inputCellMode == 1:
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
            elif inputCellMode == 2:
                batchX = np.reshape(batchX[:,bInd,:],[-1,nCell*nWindow])
            elif inputCellMode == -1:
                batchX = np.delete(batchX,bInd,1)
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
            
            w1TrainUpdate,b1TrainUpdate,_,MultilossTrain,lossb1,lossb2, predTrainb1, predTrainb2 = sess.run([w1TrainUpdate_op,b1TrainUpdate_op,trainer_class, loss_class_all,loss_class1,loss_class2, 
                                                                                   predict_class1_op, predict_class2_op], feed_dict={x:batchX, y1_class:batchY1, y2_class:batchY2})
            
            #pdb.set_trace()
            if i % 100 == 0:
                print("iteration: %d,loss of b1&b2: %f,loss of b1: %f,loss of b2: %f" % (i,MultilossTrain,lossb1,lossb2))
                  
            if i % 500 == 0:
            
                w1TestUpdate,b1TestUpdate,MultilossTest, lossTestb1, lossTestb2, predTestb1, predTestb2, accuracyb1, accuracyb2 = sess.run([w1TestUpdate_op,b1TestUpdate_op,loss_test_class_all, loss_test_class1, loss_test_class2,
                                                                                                                               predict_class1_test_op, predict_class2_test_op,
                                                                                                                               accuracy_op1,accuracy_op2], feed_dict={x:testX, y1_class:testY1,y2_class:testY2})
                print("iteration: %d,loss of b1 + b2: %f,accuracy b1: %f,accuracy b2: %f" % (i,MultilossTest,accuracyb1,accuracyb2))
            
                print("--------------------------")
                #print("predTest:\n",predTest[:10])
                print("argmax predTest of b1:",np.argmax(predTestb1[:10],axis=1))
                print("argmax testY of b1:",np.argmax(testY1[:10],axis=1))
                
                print("--------------------------")
                
                print("argmax predTest of b2:",np.argmax(predTestb1[:10],axis=1))
                print("argmax testY of b2:",np.argmax(testY2[:10],axis=1))
                print("--------------------------")
                
                # save model to file
                saver = tf.train.Saver()
                saver.save(sess,"models/modelsb{0}/model_{1}.ckpt".format(dataMode,i))
                #saver.restore(sess,"modelsb{0}/model_{1}.ckpt".format(dataMode,i))
                
                # Save loss
                with open('./visualization/lossmulti_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(MultilossTest,fp)
                    pickle.dump(lossTestb1,fp)
                    pickle.dump(lossTestb2,fp)
                # Save accuracy
                with open('./visualization/accuracymulti_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(accuracyb1,fp)
                    pickle.dump(accuracyb2,fp)

    ################# Signal Prediction ###########################
    else:
        x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow])
        y_class = tf.placeholder(tf.float32, shape=[None,nClass])
        
        predict_class_op = mytraining.nn_class(x)
        predict_class_test_op = mytraining.nn_class(x,reuse=True)
        
        #---------------------
        #---------------------
        loss_class = tf.losses.softmax_cross_entropy(y_class, predict_class_op)
        loss_test_class = tf.losses.softmax_cross_entropy(y_class, predict_class_test_op)
        
        trainer_class = tf.train.AdamOptimizer(1e-3).minimize(loss_class)
        
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_class, 1), tf.argmax(predict_class_op, 1)), tf.float32))
        
        #---------------------
        #---------------------
        # training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #---------------------
        
        if inputCellMode == 1:
            testX = np.reshape(mytraining.myData.xTest,[-1,nCell*nWindow])
        elif inputCellMode == 2:
            testX = np.reshape(mytraining.myData.xTest[:,bInd,:],[-1,nCell*nWindow])
        elif inputCellMode == -1:
            # bIndのセルを削除して、bInd以外のセルを使う
            testX = np.delete(mytraining.myData.xTest,bInd,1)
            testX = np.reshape(testX,[-1,nCell*nWindow])
        
        # eqp から渡されたbが指定されている
        testY = mytraining.myData.yTest
        
        # Start training
        Testflag=False
        for i in range(100000):
            
            batchX, batchY = mytraining.myData.nextBatch(batchSize)
            
            if inputCellMode == 1:
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
            elif inputCellMode == 2:
                batchX = np.reshape(batchX[:,bInd,:],[-1,nCell*nWindow])
            elif inputCellMode == -1:
                batchX = np.delete(batchX,bInd,1)
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
                
            # classification
            w1Train, b1Train, w2Train, b2Train, _, lossTrain, predTrain = sess.run([mytraining.w1, mytraining.b1, mytraining.w2, mytraining.b2, trainer_class, loss_class, predict_class_op], feed_dict={x:batchX, y_class:batchY})
        
            if i % 100 == 0:
                print("iteration: %d,loss: %f" % (i,lossTrain))
                
            if i % 500 == 0:
                
                w1Test, b1Test, w2Test, b2Test, lossTest, predTest, accuracy  = sess.run([mytraining.w1, mytraining.b1, mytraining.w2, mytraining.b2, loss_test_class, predict_class_test_op, accuracy_op], feed_dict={x:testX, y_class:testY})
                
                if not Testflag:
                    minw1Test = w1Test
                    minw1Train = w1Train
                    
                    minb1Test = b1Test
                    minb1Train = b1Train
                    
                    minw2Test = w2Test
                    minw2Train = w2Train
                    
                    
                    minb2Test = b2Test
                    minb2Train = b2Train
                    
                    #minTestloss = lossTest
                    maxaccuracy = accuracy
                    Testflag = True
                    
                else:
                    if accuracy > maxaccuracy:
                        
                        minw1Test = w1Test
                        minw1Train = w1Train
                        
                        minb1Test = b1Test
                        minb1Train = b1Train
                        #w2(b2)もほしい時
                        minw2Test = w2Test
                        minw2Train = w2Train
                        
                        minb2Test = b2Test
                        minb2Train = b2Train
                        
                        maxaccuracy = accuracy
                        
                print("iteration: %d,loss: %f, accuracy: %f" % (i,lossTest,accuracy))
            
                print("--------------------------")
                #print("predTest:\n",predTest[:10])
                print("argmax predTest:",np.argmax(predTest[:10],axis=1))
                print("argmax testY:",np.argmax(testY[:10],axis=1))
                print("--------------------------")

                
                # Save loss
                # loss　もaccuracy　も番号を変える
                with open('./visualization/lossb1_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(lossTest,fp)
                # Save accuracy
                with open('./visualization/accuracyb1_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(accuracy,fp)


        # w1,b1を保存
        with open("trainingw{0}/process_{1}.pickle".format(outputCellMode,i), "wb") as fp:
                    pickle.dump(minw1Train,fp)
                    pickle.dump(minw1Test,fp)
                    pickle.dump(minb1Train,fp)
                    pickle.dump(minb1Test,fp)

                    
                
                
                
        

#-*- coding: utf-8 -*-
"""
Created on Wed May 30 17:32:45 2018

@author: yu
"""

# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from scipy.spatial.distance import correlation
from scipy.stats import pearsonr
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
            
        
        # b1単独予測器の場合は_1指定
        # b2単独予測器の場合は_2指定
        # b1b2組み合わせデータの場合は_12指定

        if dataMode == 12:
            dataPath = 'b1b2'
            picklePath='listxydatab1b2_12.pkl'
            trainingpicklePath='listtraintestdatab1b2_12.pkl'
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
        self.nYear = 8000
        # b1+b2+hidden層(回帰の入力)
        self.b1b2Hidden = 482
        # b1+b2+特徴量(回帰の入力)
        self.b1b2X = 82
        
        self.zdims = 8    
        self.cell = 1
        self.yvdataPath = 'b1b2_yV'
        
        
        self.myData = eqp.Data(fname=fname, trainRatio=0.8, nCell=nCell,
                               sYear=2000, bInd=bInd, eYear=10000, 
                               isWindows=True, isClass=True, 
              dataMode=dataMode, outputCellMode=outputCellMode, datapickleMode=datapickleMode,
              featuresPath=self.featuresPath,dataPath=dataPath,
              trainingpicklePath=trainingpicklePath,picklePath=picklePath)

    def weight_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    
    def bias_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
            
    def conv1d_relu(self,inputs, w, b, stride):
        #filter: [kernel, output_depth, input_depth]
        conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
        conv = tf.nn.relu(conv)
        return conv
    
    def conv1d_t_relu(self,inputs, w, b, output_shape, stride):
        conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
        conv = tf.nn.relu(conv)
        return conv
    
    def softmax(self,inputs,w,b,keepProb):
         softmax = tf.matmul(inputs,w) + b
         softmax = tf.nn.dropout(softmax, keepProb)
         softmax = tf.nn.softmax(softmax)
         return softmax
        
    def fc_relu(self,inputs, w, b, keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu
        

    def fc(self, inputs,w, b, keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc
        
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def CNN(self,yv,reuse=False):
        
        with tf.variable_scope('cnn') as scope:
            if reuse:
                scope.reuse_variables()
        
            # 8000 -> 800
            # 1layer
            convW1 = self.weight_variable("convW1", [10, 1, 32])
            convB1 = self.bias_variable("convB1", [32])
            conv1 = self.conv1d_relu(yv, convW1, convB1, stride=10)
            
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
            fcW2 = self.weight_variable("fcW2", [128, self.zdims])
            fcB2 = self.bias_variable("fcB2", [self.zdims])
            fc2 = self.fc_relu(fc1, fcW2, fcB2)
            
            return fc2

    def multiCellnn(self,x,reuse=False):
        with tf.variable_scope('multiCellnn') as scope:  
            keepProb = 0.7
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
             
            #input -> hidden
            #学習させない時は、trainable=False(上書き値で固定)
            self.multiw1 = self.weight_variable('multiw1',[self.nCell*self.nWindow,self.multinHidden])
            self.multibias1 = self.bias_variable('bias1',[self.multinHidden])
            multih = self.fc_relu(x,self.multiw1,self.multibias1,keepProb) 
            
            #hidden -> output
            w2_1 = self.weight_variable('w2_1',[self.multinHidden,self.nClass])
            bias2_1 = self.bias_variable('bias2_1',[self.nClass])
            
            w2_2 = self.weight_variable('w2_2',[self.multinHidden,self.nClass])
            bias2_2 = self.bias_variable('bias2_2',[self.nClass])
            
            
            y1 = self.fc(multih,w2_1,bias2_1,keepProb)
            y2 = self.fc(multih,w2_2,bias2_2,keepProb)
           
            
            return y1, y2
    
    def regression(self,x,reuse=False):
        
        with tf.variable_scope('regression') as scope:  
            keepProb = 1
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            
            w1_regression = self.weight_variable('w1_regression',[self.b1b2X,128])
            bias1_regression = self.bias_variable('bias1_regression',[128])
            
            fc1 = self.fc_relu(x,w1_regression,bias1_regression,keepProb)
            
            w2_regression = self.weight_variable('w2_regression',[128,2])
            bias2_regression = self.bias_variable('bias2_regression',[2])
            
            y = self.fc(fc1,w2_regression,bias2_regression,keepProb)
            
            return y

#---------------------
#---------------------
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
    nYear = mytraining.nYear
    batchSize = 7000
    multinHidden = mytraining.multinHidden
    b_b12ClassNormalization = 0.00575
    b_b3ClassNormalization = 0.0017
    zdims = mytraining.zdims
    yvdataPath = mytraining.yvdataPath
    # yvb が入っているpath
    yvfullPath = os.path.join(featuresPath,yvdataPath,'yvb.pickle')
    
    # b1b2両方を出力したいときは True
    ismultiCellnn = True
    
    if ismultiCellnn:
        ################# CNN ##################
        rawyV1,rawyV2,rawyV3,rawyV4,rawyV5,rawyV6,rawyV7,rawyV8 = tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear]),tf.placeholder(tf.float32,shape=[None,nYear])
        
        # CNN 特徴量
        cnnyv1_op,cnnyv2_op,cnnyv3_op,cnnyv4_op,cnnyv5_op,cnnyv6_op,cnnyv7_op,cnnyv8_op = mytraining.CNN(rawyV1),mytraining.CNN(rawyV2),mytraining.CNN(rawyV3),mytraining.CNN(rawyV4),mytraining.CNN(rawyV5),mytraining.CNN(rawyV6),mytraining.CNN(rawyV7),mytraining.CNN(rawyV8)
        cnnyv1_test_op,cnnyv2_test_op,cnnyv3_test_op,cnnyv4_test_op,cnnyv5_test_op,cnnyv6_test_op,cnnyv7_test_op,cnnyv8_test_op = mytraining.CNN(rawyV1,reuse=True),mytraining.CNN(rawyV2,reuse=True),mytraining.CNN(rawyV3,reuse=True),mytraining.CNN(rawyV4,reuse=True),mytraining.CNN(rawyV5,reuse=True),mytraining.CNN(rawyV6,reuse=True),mytraining.CNN(rawyV7,reuse=True),mytraining.CNN(rawyV8,reuse=True)
        
        pdb.set_trace()

        cnn_yv = tf.concat((tf.expand_dims(cnnyv1_op,1),tf.expand_dims(cnnyv2_op,1),tf.expand_dims(cnnyv3_op,1),tf.expand_dims(cnnyv4_op,1),tf.expand_dims(cnnyv5_op,1),tf.expand_dims(cnnyv6_op,1),tf.expand_dims(cnnyv7_op,1),tf.expand_dims(cnnyv8_op,1)),1)
        #cnn_yv = tf.reshape(cnn_yv,[:,-1])

        ########### Classification ##################        
        x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow]) 
        y1_class = tf.placeholder(tf.float32, shape=[None,nClass])
        y2_class = tf.placeholder(tf.float32, shape=[None,nClass])
        y1_class_label = tf.placeholder(tf.float32, shape=[None,1])
        y2_class_label = tf.placeholder(tf.float32, shape=[None,1]) 
        
        # 12クラス分類 
        predict_class1_op,predict_class2_op = mytraining.multiCellnn(x)
        predict_class1_test_op,predict_class2_test_op = mytraining.multiCellnn(x,reuse=True)
        
        
        loss_class1 = tf.losses.softmax_cross_entropy(y1_class, predict_class1_op)
        loss_test_class1 = tf.losses.softmax_cross_entropy(y1_class, predict_class1_test_op)
    
        loss_class2 = tf.losses.softmax_cross_entropy(y2_class, predict_class2_op)
        loss_test_class2 = tf.losses.softmax_cross_entropy(y2_class, predict_class2_test_op)
        
        # loss of b1 + b2
        loss_class_all = loss_class1 + loss_class2
        loss_test_class_all = loss_test_class1 + loss_test_class2 

        loss_class_all_test = loss_test_class1 + loss_test_class2
        loss_test_class_all_test = loss_test_class1 + loss_test_class2 
        

        # training by classification
        trainer_class = tf.train.AdamOptimizer(1e-3).minimize(loss_class_all)
        
        # accuracy Test
        accuracy_test_op1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y1_class, 1), tf.argmax(predict_class1_test_op, 1)), tf.float32))
        accuracy_test_op2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y2_class, 1), tf.argmax(predict_class2_test_op, 1)), tf.float32))
         
        ################## Regression #########################
        b1b2_residual_class = tf.placeholder(tf.float32,shape=[None,2]) 
        b1b2_residual_class_test = tf.placeholder(tf.float32,shape=[None,2]) 
        
        # 最大クラス
        pred_class1_maxclass,pred_class2_maxclass = tf.argmax(predict_class1_op,axis=1),tf.argmax(predict_class2_op,axis=1) 
        pred_class1_maxclass,pred_class2_maxclass = tf.expand_dims(tf.cast(pred_class1_maxclass,tf.float32),1),tf.expand_dims(tf.cast(pred_class2_maxclass,tf.float32),1)  
        pred_class1_maxclass_test,pred_class2_maxclass_test = tf.argmax(predict_class1_test_op,axis=1),tf.argmax(predict_class2_test_op,axis=1) 
        pred_class1_maxclass_test,pred_class2_maxclass_test = tf.expand_dims(tf.cast(pred_class1_maxclass_test,tf.float32),1),tf.expand_dims(tf.cast(pred_class2_maxclass_test,tf.float32),1) 
        
        # 予測したクラスを見て、クラスの中心値を代入する       
        predict_class1_max_center_op,predict_class2_max_center_op = pred_class1_maxclass*0.0005+0.01125,pred_class2_maxclass*0.0005+0.01125
        predict_class1_max_center_test_op,predict_class2_max_center_test_op = pred_class1_maxclass_test*0.0005+0.01125,pred_class2_maxclass_test*0.0005+0.01125
        
        # 真値残差(b-b')      
        b1_residual,b2_residual = y1_class_label-predict_class1_max_center_op, y2_class_label-predict_class2_max_center_op 
        b1b2_residual = tf.concat((b1_residual,b2_residual),axis=1)
         
        b1_residual_test,b2_residual_test = y1_class_label-predict_class1_max_center_test_op, y2_class_label-predict_class2_max_center_test_op 
        b1b2_residual_test = tf.concat((b1_residual_test,b2_residual_test),axis=1)
        
        # b-b':正規化する範囲を狭くするときに使用
        b1b2_residual_normalization = (b1b2_residual_class+0.00575)/(0.00575*2)   
        b1b2_residual_normalization_test = (b1b2_residual_class_test+0.00575)/(0.00575*2)   
        
        
        # 中央値
        predict_center_class1_class2 = tf.concat((predict_class1_max_center_op,predict_class2_max_center_op),axis=1)
        predict_center_class1_class2_test = tf.concat((predict_class1_max_center_test_op,predict_class2_max_center_test_op),axis=1)
        
        # 中央値と特徴量
        predict_center_class_x =  tf.concat((predict_center_class1_class2,x),axis=1)
        predict_center_class_x_test =  tf.concat((predict_center_class1_class2_test,x),axis=1)
        
        predict_regression_residual_op= mytraining.regression(predict_center_class_x)
        predict_regression_residual_test_op= mytraining.regression(predict_center_class_x_test,reuse=True)
         
        # regressionの出力にexp にして(logスケールにする)
        exp_regression = tf.exp(predict_regression_residual_op) 
        exp_regression_test = tf.exp(predict_regression_residual_test_op)
        
        # loss(abs)
        loss_residual = tf.reduce_mean((tf.abs(exp_regression-b1b2_residual_normalization)))        
        loss_residual_test = tf.reduce_mean((tf.abs(exp_regression_test-b1b2_residual_normalization_test)))        
        
        # classification flozen
        regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="regression") 
        trainer_regression = tf.train.AdamOptimizer(1e-3).minimize(loss_residual,var_list=regressionVars)
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        
        # training
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        #---------------------

        #---------------------
        # cellの数指定
        if inputCellMode == 1:
            testX = np.reshape(mytraining.myData.xTest,[-1,nCell*nWindow])
        elif inputCellMode == 2:
            testX = np.reshape(mytraining.myData.xTest[:,bInd,:],[-1,nCell*nWindow])
        elif inputCellMode == -1:
            # bIndのセルを削除して、bInd以外のセルを使う
            testX = np.delete(mytraining.myData.xTest,bInd,1)
            testX = np.reshape(testX,[-1,nCell*nWindow])
        elif inputCellMode == 8:
            testX1,testX2,testX3,testX4,testX5,testX6,testX7,testX8 = mytraining.myData.xTest[:,0,:],mytraining.myData.xTest[:,1,:],mytraining.myData.xTest[:,2,:],mytraining.myData.xTest[:,3,:],mytraining.myData.xTest[:,4,:],mytraining.myData.xTest[:,5,:],mytraining.myData.xTest[:,6,:],mytraining.myData.xTest[:,7,:]

        # eqp から渡されたbが指定されている
        testY1 = mytraining.myData.y1Test
        testY2 = mytraining.myData.y2Test
        
        testY1Label = mytraining.myData.y1TestLabel
        testY2Label = mytraining.myData.y2TestLabel
        
        testY1Label = testY1Label[:,np.newaxis]
        testY2Label = testY2Label[:,np.newaxis]
        
        # Start training
        for i in range(100000):
             
            batchX, batchY1,batchY1Label,batchY2,batchY2Label = mytraining.myData.nextBatch(batchSize)
             
            batchY1Label = batchY1Label[:,np.newaxis]
            batchY2Label = batchY2Label[:,np.newaxis]
            
            if inputCellMode == 1:
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
                testX = np.reshape(testX,[-1,nCell*nWindow])
            elif inputCellMode == 2:
                batchX = np.reshape(batchX[:,bInd,:],[-1,nCell*nWindow])
            elif inputCellMode == -1:
                batchX = np.delete(batchX,bInd,1)
                batchX = np.reshape(batchX,[-1,nCell*nWindow])
            elif inputCellMode == 8:
                batchX1,batchX2,batchX3,batchX4,batchX5,batchX6,batchX7,batchX8 = batchX[:,0,nYear],batchX[:,1,nYear],batchX[:,2,nYear],batchX[:,3,nYear],batchX[:,4,nYear],batchX[:,5,nYear],batchX[:,6,nYear],batchX[:,7,nYear]
            
            cnn_yV1,cnn_yV2,cnn_yV3,cnn_yV4,cnn_yV5,cnn_yV6,cnn_yV7,cnn_yV7=sess.run(cnnyv1_op,cnnyv2_op,cnnyv3_op,cnnyv4_op,cnnyv5_op,cnnyv6_op,cnnyv7_op,cnnyv8_op,feed_dict={rawyV1:batchX1,rawyV2:batchX2,rawyV3:batchX3,rawyV4:batchX4,rawyV5:batchX5,rawyV6:batchX6,rawyV7:batchX7,rawyV8:batchX8})
            # classification
            _,MultilossTrain,lossb1,lossb2,predTrainb1,predTrainb2,b1residual,b2residual,b1b2residual = sess.run([trainer_class, loss_class_all,loss_class1,loss_class2,predict_class1_op, predict_class2_op,
                b1_residual,b2_residual,b1b2_residual], feed_dict={x:batchX, y1_class:batchY1, y2_class:batchY2,y1_class_label:batchY1Label, y2_class_label:batchY2Label}) 
               
            # 正規化する範囲を狭くする(b-b') 
            b1_r3,b2_r3 = np.where(b1residual<-0.00575,-0.00575,np.where(b1residual>0.00575,0.00575,b1residual)),np.where(b2residual<-0.00575,-0.00575,np.where(b2residual>0.00575,0.00575,b2residual))
            b1b2ResidualClass = np.concatenate((b1_r3,b2_r3),1)
            
            
            # regression(範囲を狭く)
            _,lossResidual= sess.run([trainer_regression,loss_residual],feed_dict={x:batchX,b1b2_residual_class:b1b2ResidualClass})
                 
            if i % 100 == 0: 
                    
                print("------------------------------------")
                print("iteration: %d,loss of b1&b2: %f,loss of b1: %f,loss of b2: %f" % (i,MultilossTrain,lossb1,lossb2))
                print("regression loss %f" %(lossResidual))
                print("------------------------------------")
                    
                """
                with open('./visualization/loss/Trainclassloss_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(MultilossTrain,fp)
                """
                
                with open('./visualization/loss/Trainclassregression_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(lossResidual,fp)
                
            # Test
            if i % 500 == 0:
                # classification                    
                MultilossTest,lossTestb1,lossTestb2,accuracyb1,accuracyb2,predCenterTest1,predCenterTest2,b1residual_test,b2residual_test,predTestb1,predTestb2 = sess.run([loss_test_class_all,loss_test_class1,loss_test_class2,accuracy_test_op1,accuracy_test_op2,predict_class1_max_center_test_op,predict_class2_max_center_test_op,b1_residual_test,b2_residual_test,predict_class1_test_op, predict_class2_test_op], feed_dict={x:testX, y1_class:testY1,y2_class:testY2,y1_class_label:testY1Label, y2_class_label:testY2Label})
                
                # 正規化する範囲を狭くする(b-b') 
                b1_r3_test,b2_r3_test = np.where(b1residual_test<-0.00575,-0.00575,np.where(b1residual_test>0.00575,0.00575,b1residual_test)),np.where(b2residual_test<-0.00575,-0.00575,np.where(b2residual_test>0.00575,0.00575,b2residual_test))
                b1b2ResidualClass_test = np.concatenate((b1_r3_test,b2_r3_test),1)
                
                # regression(範囲を狭く)
                lossResidual_test,exp_r_test= sess.run([loss_residual_test,exp_regression_test],feed_dict={x:testX,b1b2_residual_class_test:b1b2ResidualClass_test})
                
                
                # 正規化を戻す(b-b'):クラス間誤差に合わせる
                exp_r_non_test = exp_r_test*(0.00575*2)-0.00575
                b1_cr,b2_cr = predCenterTest1+exp_r_non_test[:,0,np.newaxis],predCenterTest2+exp_r_non_test[:,1,np.newaxis]
                
                
            
                print("iteration: %d,loss of b1 + b2: %f,accuracy b1: %f,accuracy b2: %f" % (i,MultilossTest,accuracyb1,accuracyb2))
                
                print('b1class',np.mean(np.abs(testY1Label-predCenterTest1))) 
                print('b1regression',np.mean(np.abs(testY1Label-b1_cr))) 
                print('b2class',np.mean(np.abs(testY2Label-predCenterTest2))) 
                print('b2regression',np.mean(np.abs(testY2Label-b2_cr))) 
               
                """    
                with open('./visualization/loss/pred/TestregresionResidual_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(b1_cr,fp)
                    pickle.dump(b2_cr,fp)
                    pickle.dump(testY1Label,fp)
                    pickle.dump(testY2Label,fp)
                """
                 
                
                with open('./visualization/loss/Testclassregression_{}.pickle'.format(i),'wb') as fp:
                    pickle.dump(lossResidual_test,fp)
                
                
                
                

    ################# Signal Prediction ###########################
    ###############################################################
    
    else:    
        x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow])
        y_class = tf.placeholder(tf.float32, shape=[None,nClass])
        predict_class_op = mytraining.singleCellnn(x)
        predict_class_test_op = mytraining.singleCellnn(x,reuse=True)
        
        #---------------------
        #---------------------
        # loss
        loss_class = tf.losses.softmax_cross_entropy(y_class, predict_class_op)
        loss_test_class = tf.losses.softmax_cross_entropy(y_class, predict_class_test_op)
        
        trainer_class = tf.train.AdamOptimizer(1e-3).minimize(loss_class)
        #---------------------        
        #---------------------
        # accuracy
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_class, 1), tf.argmax(predict_class_op, 1)), tf.float32))
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
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
                
            
            w1Train, b1Train, _, lossTrain, predTrain = sess.run([mytraining.w1, mytraining.bias1, trainer_class, loss_class, predict_class_op], feed_dict={x:batchX, y_class:batchY})
        
            if i % 100 == 0:
                print("iteration: %d,loss: %f" % (i,lossTrain))
                
                # Save loss
                with open('./visualization/loss/Trainlossb{0}_{1}.pickle'.format(outputCellMode,i),'wb') as fp:
                    pickle.dump(lossTrain,fp)

            if i % 500 == 0:
                
                w1Test, b1Test, lossTest, predTest, accuracy  = sess.run([mytraining.w1, mytraining.bias1, loss_test_class, predict_class_test_op, accuracy_op], feed_dict={x:testX, y_class:testY})
                
                # 最適なW,b取り出し
                
                if not Testflag:
                    minw1Test = w1Test
                    minw1Train = w1Train
                    
                    minb1Test = b1Test
                    minb1Train = b1Train
                    
                    maxaccuracy = accuracy
                    Testflag = True
                    
                else:
                    if accuracy > maxaccuracy:
                        
                        minw1Test = w1Test
                        minw1Train = w1Train
                        
                        minb1Test = b1Test
                        minb1Train = b1Train
                        tmpi = i
                        maxaccuracy = accuracy
                
                print("iteration: %d,loss: %f, accuracy: %f" % (i,lossTest,accuracy))
        
                print("--------------------------")
                #print("testX:\n",testX[:10])
                print("--------------------------")
                #print("predTest:\n",predTest[:10])
                print("argmax predTest:",np.argmax(predTest[:10],axis=1))
                print("argmax testY:",np.argmax(testY[:10],axis=1))
                print("--------------------------")
                
                
                with open("trainingb{0}/process_{1}.pickle".format(dataMode,i), "wb") as fp:
                    pickle.dump(predTest,fp)
                    pickle.dump(myData.xTest,fp)
                    pickle.dump(myData.yTest,fp)
                    pickle.dump(lossTest,fp)
                
                
                # Save loss
                with open('./visualization/loss/lossb{0}_{1}.pickle'.format(outputCellMode,i),'wb') as fp:
                    pickle.dump(lossTest,fp)
                
                # Save accuracy
                with open('./visualization/accuracy/accuracyb{0}_{1}.pickle'.format(outputCellMode,i),'wb') as fp:
                    pickle.dump(accuracy,fp)

                # save model to file
                saver = tf.train.Saver()
                saver.save(sess,"./models/modelsb{0}/model_{1}.ckpt".format(outputCellMode,i))
        
        
        print(maxaccuracy)
        print(tmpi)
        # w1,b1を保存
        with open("./training/b1b2/trainingw{0}/process_{1}.pickle".format(outputCellMode,i), "wb") as fp:
                    pickle.dump(minw1Train,fp)
                    pickle.dump(minw1Test,fp)
                    pickle.dump(minb1Train,fp)
                    pickle.dump(minb1Test,fp)


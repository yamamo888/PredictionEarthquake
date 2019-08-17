# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:51:43 2019

@author: yu
"""
import os
import pickle
import pdb

import numpy as np

#----------------------------- paramters --------------------------------------
# training rate
trainRatio = 0.8
# number of data
nData = 8000
# number of train data
#nTrain = int(nData * trainRatio)
# number of test data
#nTest = int(nData - nTrain)

nTrain = 3200
nTest = 800
    
# batch random index
batchRandInd = np.random.permutation(nTrain)
# -----------------------------------------------------------------------------

#-----------------------------------------------------------------------------#      
def SplitTrainTest(yMin=2,yMax=6,pNum=5,noise=0):
    """
    Split data into train data and test data.
    
    Args:
        yMin:
        yMax:
        pNum: Rotation of explanatory variable x1, x2
        noise:
        nTrain: Number of train data
    Returns:
        train,test data
    """
    
    
    # nTrain,nTest:1000,7000 3200,800 500,3500 6400,1600
    sigma = 0.0000001
    pNum = 2
    dataPath = "trainData"
    toydataPath = "toyData_{}_{}_{}_{}.pickle".format(sigma,pNum,nTrain,nTest)
    toyDatafullPath = os.path.join(dataPath,toydataPath)
    with open(toyDatafullPath,"rb") as fp:
        x1Train = pickle.load(fp)
        x2Train = pickle.load(fp)
        yTrain = pickle.load(fp)
        x1Test = pickle.load(fp)
        x2Test = pickle.load(fp)
        yTest = pickle.load(fp)
        y = pickle.load(fp)
    
    """
    # Make target variable, y ~ U(x) U: i.i.d.
    y = np.random.uniform(yMin,yMax,nData)
    x1 = np.sin(pNum * y) + 1 / np.log(y) + noise
    x2 = np.cos(pNum * y) + np.log(y) + noise
    
    # split all data to train & test data
    x1Train = x1[:nTrain][:,np.newaxis]
    x2Train = x2[:nTrain][:,np.newaxis]
    yTrain = y[:nTrain][:,np.newaxis]
    x1Test = x1[nTrain:][:,np.newaxis]
    x2Test = x2[nTrain:][:,np.newaxis]
    yTest = y[nTrain:][:,np.newaxis]
    """
    # shape=[number of data, dimention]
    return x1Train, x2Train, yTrain, x1Test, x2Test, yTest, y
#-----------------------------------------------------------------------------#      
def AnotationY(target,yMin=2,yMax=6,nClass=10,beta=1):
    """
    Anotate target variables y.
    """
    
    # class
    yClass = np.arange(yMin,yMax + beta, beta) 
 
    flag = False
    for nInd in np.arange(target.shape[0]):
        tmpY = target[nInd]
        oneHot = np.zeros(len(yClass))
        ind = 0
        # (最小、最大]
        for threY in yClass:
            if (tmpY > threY) & (tmpY <= threY + beta):
                      oneHot[ind] = 1            
            ind += 1
        # 最小値は0番目のクラスにする
        if target[nInd] == yMin:
            oneHot[0] = 1
        # 最大値が一番最後のクラスにラベルされるのを戻す
        if target[nInd] == yMax:
            oneHot[-2] = 1
        
        tmpY = oneHot[np.newaxis] 
              
        if not flag:
            Ylabel = tmpY
            flag = True
        else:
            Ylabel = np.vstack([Ylabel,tmpY])
            
    # 値が入っていないクラスを削除
    if len(yClass) == nClass + 1:
        Ylabel = Ylabel[:,:-1]
    
    YTrainlabel = Ylabel[:nTrain]
    YTestlabel = Ylabel[nTrain:]
    
    # shape=[number of data, number of class]
    return YTrainlabel, YTestlabel
#-----------------------------------------------------------------------------#
def nextBatch(Otr,Ttr,Tlabel,batchSize,batchCnt=0):
    """
    Make Mini Batch.
    """
    
    sInd = batchSize * batchCnt
    eInd = sInd + batchSize
    
    batchX = Otr[batchRandInd[sInd:eInd],:]
    batchY = Ttr[batchRandInd[sInd:eInd],:]
    batchlabelY = Tlabel[batchRandInd[sInd:eInd],:]
    
    if eInd + batchSize > nTrain:
        batchCnt = 0
    else:
        batchCnt += 1
    # [batchSize,number of dimention]
    return batchX,batchY,batchlabelY
#-----------------------------------------------------------------------------#
    

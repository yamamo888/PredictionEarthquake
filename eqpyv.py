# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:39:45 2018

@author: yu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:14:14 2018
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
import pickle
import pywt
import glob

import pandas as pd
import seaborn as sns
from scipy import stats

import math
import scipy.optimize

#########################################
class EarthQuakePlateModel:
        
    def __init__(self,dataMode,logName,nCell=8,nYear=10000):
        
        ############ 引数説明 ##############
        # dataMode:使うデータの入ったlogs以下のディレクトリ指定
        #   12:b1b2組み合わせデータ
        #   123:b1b2b3組み合わせデータ
        # logName:logfileに入っているlogファイル
        # nCell,nYear:このクラスで使うparameter
        #################################
        
        if dataMode == 12:
            dataPath = 'b1b2'
                
       
        
        self.logPath = './logs'
        self.dataPath = dataPath
        self.features = 'features'
    
        self.logName = logName
        self.logFullPath = os.path.join(self.logPath,self.dataPath,logName)
        
        # パラメータ
        self.nCell = nCell
        self.nYear = nYear
        self.yInd = 1
        self.vInds = [2,3,4,5,6,7,8,9]
        self.yV = np.zeros([nYear,nCell])
        
        
        
    #--------------------------

    #--------------------------
    #データの読み込み
    def loadABLV(self):
        self.data = open(self.logFullPath).readlines()
        
        # A, B, Lの取得
        self.A = np.zeros(self.nCell)
        self.B = np.zeros(self.nCell)
        self.L = np.zeros(self.nCell)
        
        for i in np.arange(1,self.nCell+1):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            self.A[i-1] = tmp[0]
            self.B[i-1] = tmp[1]
            self.L[i-1] = tmp[4]
            
        
        # Vの開始行取得
        isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
        vInd = np.where(isRTOL)[0][0]+1
        
        # Vの値の取得（vInd行から最終行まで）
        flag = False
        for i in np.arange(vInd,len(self.data)):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            
            if not flag:
                self.V = tmp
                flag = True
            else:
                self.V = np.vstack([self.V,tmp])

    #--------------------------
    
    #--------------------------
    # Vを年単位のデータに変換
    def convV2YearlyData(self,prefix):
        
        for year in np.arange(self.nYear):
            if np.sum(np.floor(self.V[:,self.yInd])==year):
                self.yV[year,:] = np.mean(self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:],axis=0)
        
        self.yV = self.yV.T
        # pklデータの保存
        self.yvdataPath = 'datab1b2_yV'
        self.yV =  self.yV[:,2000:]
        fullPath = os.path.join(self.features,self.yvdataPath,"{}{}.pkl".format(prefix,self.logName))     
        with open(fullPath,'wb') as fp:
            pickle.dump(self.yV,fp)
            pickle.dump(log.B,fp)
        
        
    #--------------------------
        
    #--------------------------
    # Vの生データのプロット
    def plotV(self,isPlotShow=False,isYearly=True,prefix='yV'):
        if isPlotShow:
            plt.close()
            fig, figInds = plt.subplots(nrows=8, sharex=True)
    
            for figInd in np.arange(len(figInds)):
                figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
                if isYearly:
                    figInds[figInd].plot(np.arange(self.nYear), self.yV[:,figInd])

            fullPath = os.path.join(self.visualPath,"{}_data.png".format(self.logName))
            plt.savefig(fullPath)
            fig.show()
            plt.show()            
        
        # pklデータの保存
        fullPath = os.path.join(self.features,self.dataPath,"{}{}.pkl".format(prefix,self.logName))
        with open(fullPath,'wb') as fp:
            pickle.dump(self.yV,fp)
            pickle.dump(log.B,fp)
            pickle.dump(self.yVkde,fp)
            pickle.dump(self.yVfft,fp)
            pickle.dump(self.X,fp)
    #--------------------------
    #------------------------------------
    
#########################################

#########################################
class Data:
    ############ 引数説明 #################################
    #fname:Class EarthQuakePlateModelで保存したpicklefileすべて
    #trainRatio:学習とテストデータの分け方
    #nCell,sYear,bInd,eYear:このクラスで使うparameter
    #isWindow:Windows用
    #isClass:classification用
    #dataMode,cellMode,datapickleMode:mainに詳細,Modeの設定
    #dataPath:pickledataが入っているディレクトリ
    #xydataPath:X,Yが保存してあるpickleファイル指定
    #traintestdataPath:TrainTestが保存してあるpickleファイル指定
    #####################################################
    
    #######[1]inputCellMode(入力のセルを選択) ######## 
    # inputCellMode=1: すべてのセル
    # inputCellMode=2: bIndのみのセル
    # inputCellMode=-1: bInd以外のセル
    ###############################################

    ####[2] dataMode(データの種類の切り替え) ##########
    # dataMode=12:b1b2
    # dataMode=123:b1b2b3
    ##################################################
        
    #####[3] cellMode（出力するセルのパラメータ[b]指定） #########
    # cellMode=1,2: b1 or b2　（1次元の出力）
    # cellMode=12: b1 b2　（２次元の出力）
    # cellMode=23: b2 b3 (2次元の出力)
    # cellMode=123: b1 b2 b3 (3次元の出力)
    #####################################################
    
    ####[4] datapickleMode （pickleが３種類）######################
    #1つめ：yV(KDE,FFTがかかっている)のデータが入ったpickle 
    #2つめ：X(=yV),Y(=cellModeで作ったｂ)のデータが入ったpickle　
    #３つめ：Train,Testのデータが入ったpickle
    
    # datapickleMode=1(3つめ): Load Train,Test data
    # datapickelMode=2(2つめ): Save Train,Test data Load X,Y
    # datapickleMode=3(１つめ): Save X,Y Load yV etc...
    ###########################################################
    
    def __init__(self,fname='yV*',trainRatio=0.8, nCell=8, 
                 sYear=2000, bInd=0, eYear=10000, isWindows=True, isClass=True, 
                 dataMode=1, outputCellMode=1, datapickleMode=1, 
                 featuresPath='./features', dataPath='datab1b2_yV',
                 trainingpicklePath='traintestdatab1.pkl',picklePath='xydatab1.pkl'):
        
        yvdataPath = 'datab1b2_yv'
         
        # pklファイルの一覧
        fullPath = os.path.join(featuresPath,yvdataPath,fname)
        files = glob.glob(fullPath)

        # データの領域確保
        self.nData = len(files)
        
        # バッチの初期化(mode=1の時のため)
        self.batchCnt = 0
        self.nTrain = np.floor(self.nData * trainRatio).astype(int)
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        yvtrainingpicklePath = 'traintestdata_yvb1b2.pkl'
        yvpicklePath = 'xydata_yvb1b2.pkl'
        
        #traintestfullPath = os.path.join(featuresPath,dataPath,trainingpicklePath)
        #xyfullPath = os.path.join(featuresPath,dataPath,picklePath)
        self.outputCellMode = outputCellMode
        
        yvtraintestfullPath = os.path.join(featuresPath,yvdataPath,yvtrainingpicklePath)
        yvxyfullPath = os.path.join(featuresPath,yvdataPath,yvpicklePath)
        
        
        if datapickleMode == 1:
            #ｘ,ｙtrain x,ytestのpickleファイル読み込み
                    
            if outputCellMode == 12 or outputCellMode == 23:
                with open(yvtraintestfullPath,'rb') as fp:
                    self.xTrain = pickle.load(fp)
                    self.y1Train = pickle.load(fp)
                    self.y2Train = pickle.load(fp)
                    self.y1TrainLabel = pickle.load(fp)
                    self.y2TrainLabel = pickle.load(fp)
                    self.xTest = pickle.load(fp)
                    self.y1Test = pickle.load(fp)
                    self.y2Test = pickle.load(fp)
                    self.y1TestLabel = pickle.load(fp)
                    self.y2TestLabel = pickle.load(fp)
                    
        elif datapickleMode == 2:
            #　XとYのpickleファイル読み込み
            #　XとYのpickleファイル読み込み
        
            if outputCellMode == 12 or outputCellMode == 23:
                with open(yvxyfullPath,'rb') as fp:
                    tmpX = pickle.load(fp)
                    listY1 = pickle.load(fp)
                    listY2 = pickle.load(fp)
                
                self.X = np.array(tmpX)
                
                # 学習データとテストデータ数
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ミニバッチの初期化
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)
            
                # ランダムにインデックスをシャッフル
                self.randInd = np.random.permutation(self.nData)

                onehotY1 = listY1[0]
                numY1 = listY1[1]
                
                onehotY2 = listY2[0]
                numY2 = listY2[1]
                
                # 学習データ
                self.xTrain = self.X[self.randInd[0:self.nTrain]]
                self.y1Train,self.y1TrainLabel = onehotY1[self.randInd[0:self.nTrain]],numY1[[self.randInd[0:self.nTrain]]] 
                self.y2Train,self.y2TrainLabel = onehotY2[self.randInd[0:self.nTrain]],numY2[[self.randInd[0:self.nTrain]]]
                
                # 評価データ
                self.xTest = self.X[self.randInd[self.nTrain:]] 
                self.y1Test,self.y1TestLabel = onehotY1[self.randInd[self.nTrain:]],numY1[[self.randInd[self.nTrain:]]] 
                self.y2Test,self.y2TestLabel = onehotY2[self.randInd[self.nTrain:]],numY2[[self.randInd[self.nTrain:]]] 
                
                pdb.set_trace()
                # 学習とテストデータの保存
                with open(yvtraintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.y1Train,fp)
                    pickle.dump(self.y2Train,fp)
                    pickle.dump(self.y1TrainLabel,fp)
                    pickle.dump(self.y2TrainLabel,fp)
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.y1Test,fp)
                    pickle.dump(self.y2Test,fp)
                    pickle.dump(self.y1TestLabel,fp)
                    pickle.dump(self.y2TestLabel,fp)
            
            
                
        elif datapickleMode == 3:
            
            # appendするリストを用意
            # データの読み込み
            listyv = []
            listy1 = []
            listy2 = []
            listtruey1 = []
            listtruey2 = []
            for fID in np.arange(self.nData):
                
                if isWindows:
                    file = files[fID].split('\\')[2]
                else:
                    file = files[fID].split('/')[2]
                    
                fullPath = os.path.join(featuresPath,yvdataPath,file)
                
                with open(fullPath,'rb') as fp:
                    tmpyV = pickle.load(fp)
                    tmpY = pickle.load(fp)
                    
                if outputCellMode == 12 or outputCellMode == 23: 
                    # pick up b in only one cell
                    tmpY1 = tmpY[bInd[0]]
                    tmpY2 = tmpY[bInd[1]]
                    
                    if isClass:
                        Y1 = tmpY[bInd[0]][np.newaxis]
                        Y2 = tmpY[bInd[1]][np.newaxis]
                        
                        sB = 0.011
                        eB = 0.0169
                        iB = 0.0005
            
                        Bs = np.arange(sB,eB,iB)
                        oneHot1 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        oneHot2 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                                        
                        ind = 0
                        for threB in Bs:
                            if (tmpY1 >= threB) & (tmpY1 < threB + iB):
                                oneHot1[ind] = 1            
                            ind += 1
                        tmpY1 = oneHot1 
                        #tmpY1 = tmpY1[np.newaxis]
                        
                        ind = 0
                        for threB in Bs:
                            if (tmpY2 >= threB) & (tmpY2 < threB + iB):
                                oneHot2[ind] = 1            
                            ind += 1
                        tmpY2 = oneHot2 
                        #tmpY2 = tmpY2[np.newaxis]
                        
                        listyv.append(tmpyV.tolist())
                        listy1.append(tmpY1.tolist())
                        listy2.append(tmpY2.tolist())
                        listtruey1.append(Y1.tolist())
                        listtruey2.append(Y2.tolist())
                        
                        listY1 = [np.array(listy1),np.array(listtruey1)] 
                        listY2 = [np.array(listy2),np.array(listtruey2)]
            #pdb.set_trace()        
            if outputCellMode == 12 or outputCellMode == 23: 
                with open(yvxyfullPath,'wb') as fp:
                    pickle.dump(listyv,fp)
                    pickle.dump(listY1,fp)
                    pickle.dump(listY2,fp)
                    
#------------------------------------
    
    #------------------------------------
    # ミニバッチの取り出し
    def nextBatch(self,batchSize,isTensorflow=True):
        
        
        if self.outputCellMode == 1 or self.outputCellMode == 2:
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
            
            
            batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
            batchY = self.yTrain[self.batchRandInd[sInd:eInd]]
            
                    
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            
            return batchX, batchY
        #------------------------------------
        #------------------------------------
        elif self.outputCellMode == 12 or self.outputCellMode == 23:
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
        
            batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
            batchY1,batchY1Label = self.y1Train[self.batchRandInd[sInd:eInd]],self.y1TrainLabel[self.batchRandInd[sInd:eInd]]
            batchY2,batchY2Label = self.y2Train[self.batchRandInd[sInd:eInd]],self.y2TrainLabel[self.batchRandInd[sInd:eInd]]
           
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            
            return batchX,batchY1,batchY1Label,batchY2,batchY2Label
        #------------------------------------
        #------------------------------------
        
            
    #------------------------------------

#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    
    isWindows = True

    # Mode 設定
    inputCellmode = int(sys.argv[1])
    dataMode = int(sys.argv[2])
    outputCellMode = int(sys.argv[3])
    datapickleMode = int(sys.argv[4])
    
    # b1単独予測器の場合は_1をつけたpickle指定 
    # b2単独予測器の場合は_2をつけたpickle指定
    # b1b2組み合わせ予測器の場合は_12をつけたpickle指定
    
    # [onehot,真値]にしたデータを使う時は、listを頭につけるpickle指定

    if dataMode == 12:
        
        dataPath = 'b1b2'   
        picklePath = 'listxydatab1b2_12.pkl'
        trainingpicklePath = 'listtraintestdatab1b2_12.pkl'
        fname = 'yV*'

            
    #ClassDataのdataMode=3でbIndを設定するのに必要
    if outputCellMode == 1:
        bInd = 0    
    elif outputCellMode == 2:
        bInd = 1
    elif outputCellMode == 12:
        bInd=[0,1]
    elif outputCellMode == 23:
        bInd=[1,2]
    elif outputCellMode == 123:
        bInd=[0,1,2]

    """
    #Reading load log.txt
    if isWindows:
        files = glob.glob('logs\\b1b2\\log_*.txt')
    else:
        files = glob.glob('./logs/b1b2/log_*.txt')
    
    for fID in np.arange(len(files)):
        print('reading',files[fID])
        if isWindows:
            #pdb.set_trace()
            file = files[fID].split('\\')[2]
            
        else:
            file = files[fID].split('/')[3]
        
        # 地震プレートモデル用のオブジェクト
        #log = EarthQuakePlateModel(dataMode,file,nCell=8,nYear=10000)
        #log.loadABLV()
        #log.convV2YearlyData(prefix='yV')
        
        
        ############# Data作成の手順 #########################
        # 1. data=...の中のMode,Pathを指定する
        # 2. datapickleMode　を　3-2-1の順番でpickleを保存と読み込み
        #####################################################
        
    """
    #このファイルから直接Mode指定するときは、dataインスタンスの値を変更する必要がある（コマンドからは反応しない）
    data = Data(fname=fname,trainRatio=0.8, nCell=8, 
                sYear=2000, bInd=bInd, eYear=10000, isWindows=isWindows, isClass=True,
                dataMode=dataMode, outputCellMode=outputCellMode, 
                datapickleMode=datapickleMode,featuresPath='features', dataPath=dataPath,
                trainingpicklePath=trainingpicklePath,picklePath=picklePath)
        
    
        
    
############################################
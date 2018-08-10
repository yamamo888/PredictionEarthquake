# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 00:12:10 2018

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
                
        elif dataMode == 123:
            dataPath = 'b1b2b3'

        # 組み合わせデータの保存場所
        self.logPath = './logs'
        self.dataPath = dataPath
        
        # log file
        self.logName = logName
        # log file Path
        self.logFullPath = os.path.join(self.logPath,self.dataPath,logName)
        # parameters
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
    def convV2YearlyData(self):
        
        for year in np.arange(self.nYear):
            if np.sum(np.floor(self.V[:,self.yInd])==year):
                self.yV[year,:] = np.mean(self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:],axis=0)
        
        self.yV = self.yV.T
        
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
        fullPath = os.path.join(self.dataPath,"{}{}.pkl".format(prefix,self.logName))
        with open(fullPath,'wb') as fp:
            pickle.dump(self.yV,fp)
            pickle.dump(log.B,fp)
            pickle.dump(self.yVkde,fp)
            pickle.dump(self.yVfft,fp)
            pickle.dump(self.X,fp)
    #--------------------------
    

    #------------------------------------
    # イベントデータ（デルタ関数）を、KDEで滑らかにつなげる
    def KDE(self, v_divid = 10.0, bw = 0.01):

        flag = False
        for cellInd in np.arange(self.nCell):

            #　速度vから地震が何回起きたか相対的に取得
            v_width = self.yV[cellInd,:].max() / v_divid
            eqp_num = np.floor(self.yV[cellInd,:] / v_width)
                
            # list(float) -> list(int) -> array
            eqp_tmp = list(map(int,eqp_num))
            eqp_num = np.array(eqp_tmp)

            # 年数を任意の回数増やした(0回のデータは消える)
            eqp_v = np.repeat(np.arange(0,self.nYear),eqp_num)
            
            # KDE
            x_grid = np.arange(0,self.nYear)
            kde_model = stats.gaussian_kde(eqp_v,bw_method=bw)
            kde_model = kde_model(x_grid)
            kde_model = kde_model[np.newaxis,:]

            if not flag:
                self.yVkde = kde_model
                flag = True
            else:
                self.yVkde = np.concatenate((self.yVkde,kde_model),axis=0)
    #--------------------------

    #--------------------------
    # 周波数特徴量の抽出
    def FFT(self,widthWindow=25,eFrq=250, sYear=2000, eYear=10000):

        # FFTの計算
        self.yVfft = np.abs(np.fft.fft(self.yVkde[:,sYear:eYear]))

        #----------------------
        # スペクトラムをスライディングウィンドウごとに平均をとった特徴量の抽出
        flag = False
        for cnt in np.arange(int(eFrq/widthWindow)):
            sInd = widthWindow * cnt + 1
            eInd = sInd + widthWindow
            
            # ウィンドウのスペクトラムの平均(周波数スペクトル)（ピークに反応できない場合）
            #平均や最小値をとったりする（次元数を増やす必要がある）
            #X = np.mean(self.yVfft[:,sInd:eInd],axis=1)
            X = np.max(self.yVfft[:,sInd:eInd],axis=1)
            X = X[np.newaxis]

            if not flag:
                self.X = X
                flag = True
            else:
                self.X = np.concatenate((self.X,X),axis=0)

        self.X = self.X.T
    #--------------------------
    
#########################################

#########################################
class Data:
    
    #-------------------------
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
    
    
    def __init__(self,fname='kde_fft_log_10*',trainRatio=0.8, nCell=8, 
                 sYear=2000, bInd=0, eYear=10000, isWindows=True, isClass=False,
                 dataMode=1, outputCellMode=1, datapickleMode=1,featuresPath='./features', dataPath='datab1',
                 trainingpicklePath='traintestdatab1.pkl',picklePath='xydatab1.pkl'):
        
        
        # pklファイルの一覧
        fullPath = os.path.join(featuresPath,dataPath,fname)
        files = glob.glob(fullPath)

        # データの領域確保
        self.nData = len(files)
        
        # バッチの初期化(mode=1の時のため)
        self.batchCnt = 0
        self.nTrain = np.floor(self.nData * trainRatio).astype(int)
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        
        traintestfullPath = os.path.join(featuresPath,dataPath,trainingpicklePath)
        xyfullPath = os.path.join(featuresPath,dataPath,picklePath)
        self.outputCellMode = outputCellMode
        
        
        if datapickleMode == 1:
            #ｘ,ｙtrain x,ytestのpickleファイル読み込み
            if outputCellMode == 1 or outputCellMode ==2:
                with open(traintestfullPath,'rb') as fp:
                    self.xTrain = pickle.load(fp)
                    self.yTrain = pickle.load(fp)
                    self.xTest = pickle.load(fp)
                    self.yTest = pickle.load(fp)
            
            elif outputCellMode == 12 or outputCellMode == 23:
                with open(traintestfullPath,'rb') as fp:
                    self.xTrain = pickle.load(fp)
                    self.y1Train = pickle.load(fp)
                    self.y2Train = pickle.load(fp)
                    self.xTest = pickle.load(fp)
                    self.y1Test = pickle.load(fp)
                    self.y2Test = pickle.load(fp)
            
            elif outputCellMode == 123:
                with open(traintestfullPath,'rb') as fp:
                    self.xTrain = pickle.load(fp)
                    self.y1Train = pickle.load(fp)
                    self.y2Train = pickle.load(fp)
                    self.y3Train = pickle.load(fp)
                    self.xTest = pickle.load(fp)
                    self.y1Test = pickle.load(fp)
                    self.y2Test = pickle.load(fp)
                    self.y3Test = pickle.load(fp)

        elif datapickleMode == 2:
        
            #　XとYのpickleファイル読み込み
            if outputCellMode == 1 or outputCellMode == 2:
                with open(xyfullPath,'rb') as fp:
                    self.X = pickle.load(fp)
                    self.Y = pickle.load(fp)
                
                """
                # XとYの正規化(Yが小さすぎるため,そのもののbを可視化したいときはYの正規化だけはずす）
                self.minY = np.min(self.Y)
                self.maxY = np.max(self.Y)
                self.Y = (self.Y - self.minY)/(self.maxY-self.minY)
                
                self.minX = np.min(self.X)
                self.maxX = np.max(self.X)
                self.X = (self.X - self.minX)/(self.maxX-self.minX)
                self.X = (self.X-np.mean(self.X,axis=0))*100
                self.Y = self.Y * 100 - 1
                """
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ミニバッチの初期化
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)
            
                # 学習データとテストデータ数
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ランダムにインデックスをシャッフル
                self.randInd = np.random.permutation(self.nData)
                
                
                # 学習データ
                self.xTrain = self.X[self.randInd[0:self.nTrain]]
                self.yTrain = self.Y[self.randInd[0:self.nTrain]]
                
                # 評価データ
                self.xTest = self.X[self.randInd[self.nTrain:]]
                self.yTest = self.Y[self.randInd[self.nTrain:]]
            
                # 学習とテストデータの保存
                with open(traintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.yTrain,fp)
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.yTest,fp)
            
            
            elif outputCellMode == 12 or outputCellMode == 23:
                with open(xyfullPath,'rb') as fp:
                    self.X = pickle.load(fp)
                    self.Y1 = pickle.load(fp)
                    self.Y2 = pickle.load(fp)
                
                """
                # XとYの正規化(Yが小さすぎるため,そのもののbを可視化したいときはYの正規化だけはずす）
                self.minY = np.min(self.Y)
                self.maxY = np.max(self.Y)
                self.Y = (self.Y - self.minY)/(self.maxY-self.minY)
                
                self.minX = np.min(self.X)
                self.maxX = np.max(self.X)
                self.X = (self.X - self.minX)/(self.maxX-self.minX)
                self.X = (self.X-np.mean(self.X,axis=0))*100
                self.Y = self.Y * 100 - 1
                """
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ミニバッチの初期化
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)
            
                # 学習データとテストデータ数
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ランダムにインデックスをシャッフル
                self.randInd = np.random.permutation(self.nData)
                
                # 学習データ
                self.xTrain = self.X[self.randInd[0:self.nTrain]]
                self.y1Train = self.Y1[self.randInd[0:self.nTrain]]
                self.y2Train = self.Y2[self.randInd[0:self.nTrain]]
                
                # 評価データ
                self.xTest = self.X[self.randInd[self.nTrain:]] 
                self.y1Test = self.Y1[self.randInd[self.nTrain:]]
                self.y2Test = self.Y2[self.randInd[self.nTrain:]]
                
                # 学習とテストデータの保存
                with open(traintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.y1Train,fp)
                    pickle.dump(self.y2Train,fp)
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.y1Test,fp)
                    pickle.dump(self.y2Test,fp)
            
            elif outputCellMode == 123:
                with open(xyfullPath,'rb') as fp:
                    self.X = pickle.load(fp)
                    self.Y1 = pickle.load(fp)
                    self.Y2 = pickle.load(fp)
                    self.Y3 = pickle.load(fp)

                self.nTrain = np.floor(seld.nData,trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)

                self.xTrain = self.X[self.randInd[0:sef.nTrain]]
                self.y1Train = self.Y1[self.randInd[0:self.nTrain]]
                self.y2Train = self.Y2[self.randInd[0:self.nTrain]] 
                self.y3Train = self.Y3[self.randInd[0:self.nTrain]]

                self.xTest = self.X[self.randInd[self.nTrain:]]
                self.y1Test = self.Y1[self.randInd[self.nTrain:]]
                self.y2Test = self.Y2[self.randInd[self.nTrain:]]
                self.y3Test = self.Y3[self.randInd[self.nTrain:]]

                with open(traintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.y1Train,fp)
                    pickle.dump(self.y2Train,fp)
                    pickle.dump(self.y3Train,fp)
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.y2Test,fp)
                    pickle.dump(self.y1Test,fp)
                    pickle.dump(self.y3Test,fp)


        elif datapickleMode == 3:
            flag = False
            # データの読み込み
            for fID in np.arange(self.nData):
                
                if isWindows:
                    file = files[fID].split('\\')[1]
                else:
                    file = files[fID].split('/')[1]
                    
                fullPath = os.path.join(dataPath,file)
                
                with open(fullPath,'rb') as fp:
                    tmpyV = pickle.load(fp)
                    tmpY = pickle.load(fp)
                    tmpyVkde = pickle.load(fp)
                    tmpyVfft = pickle.load(fp)
                    tmpX = pickle.load(fp)
                
                if outputCellMode == 1 or outputCellMode ==2:
                    # pick up b in only one cell
                    
                    tmpY = tmpY[bInd]
                    if isClass:
                        sB = 0.011
                        eB = 0.0169
                        iB = 0.0005
            
                        Bs = np.arange(sB,eB,iB)
                        oneHot = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                                        
                        ind = 0
                        for threB in Bs:
                            if (tmpY >= threB) & (tmpY < threB + iB):
                                oneHot[ind] = 1            
                            ind += 1
                        tmpY = oneHot 
                        tmpY = tmpY[np.newaxis]
                        
                    if not flag:
                        self.X = tmpX[np.newaxis]
                        self.Y = tmpY
                        flag = True
                    else:
                        self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
                        self.Y = np.append(self.Y, tmpY,axis=0)
               
                elif outputCellMode == 12 or outputCellMode == 23: 
                    # pick up b in only one cell
                    tmpY1 = tmpY[bInd[0]]
                    tmpY2 = tmpY[bInd[1]]
                    
                    if isClass:
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
                        tmpY1 = tmpY1[np.newaxis]
                        
                        ind = 0
                        for threB in Bs:
                            if (tmpY2 >= threB) & (tmpY2 < threB + iB):
                                oneHot2[ind] = 1            
                            ind += 1
                        tmpY2 = oneHot2 
                        tmpY2 = tmpY2[np.newaxis]
                        
                        
                    if not flag:
                        self.X = tmpX[np.newaxis]
                        self.Y1 = tmpY1
                        self.Y2 = tmpY2
                        
                        flag = True
                    else:
                        self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
                        self.Y1 = np.append(self.Y1, tmpY1,axis=0)
                        self.Y2 = np.append(self.Y2, tmpY2,axis=0)
               
                elif outputCellMode == 123:

                    tmpY1 = tmpY[bInd[0]]
                    tmpY2 = tmpY[bInd[1]]
                    tmpY3 = tmpY[bInd[2]]
                  
                    if isClass:
                        sB = 0.011
                        eB = 0.0169
                        iB = 0.0005
            
                        Bs = np.arange(sB,eB,iB)
                        oneHot1 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        oneHot2 = np.zeros(len(Bs))#0.001,0.0015,...0.00165 
                        oneHot3 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                       
                        #b1
                        ind = 0
                        for threB in Bs:
                            if (tmpY1 >= threB) & (tmpY1 < threB + iB):
                                oneHot1[ind] = 1            
                            ind += 1
                        tmpY1 = oneHot1 
                        tmpY1 = tmpY1[np.newaxis]
                        
                        #b2
                        ind = 0
                        for threB in Bs:
                            if (tmpY2 >= threB) & (tmpY2 < threB + iB):
                                oneHot2[ind] = 1            
                            ind += 1
                        tmpY2 = oneHot2 
                        tmpY2 = tmpY2[np.newaxis]
                        
                        #b3
                        ind = 0
                        for threB in Bs:
                            if (tmpY3 >= threB) & (tmpY3 < threB + iB):
                                oneHot3[ind] = 1            
                            ind += 1
                        tmpY3 = oneHot3 
                        tmpY3 = tmpY3[np.newaxis]
                        
                    if not flag:
                        self.X = tmpX[np.newaxis]
                        self.Y1 = tmpY1
                        self.Y2 = tmpY2
                        self.Y3 = tmpY3
                        flag = True
                    else:
                        self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
                        self.Y1 = np.append(self.Y1, tmpY1,axis=0)
                        self.Y2 = np.append(self.Y2, tmpY2,axis=0)
                        self.Y3 = np.append(self.Y3, tmpY3,axis=0)
            
            # X,Y の保存
            if outputCellMode ==1 or outputCellMode == 2:
                with open(xyfullPath,'wb') as fp:
                    pickle.dump(self.X,fp)
                    pickle.dump(self.Y,fp)
            
            
            elif outputCellMode == 12 or outputCellMode == 23: 
                with open(xyfullPath,'wb') as fp:
                    pickle.dump(self.X,fp)
                    pickle.dump(self.Y1,fp)
                    pickle.dump(self.Y2,fp)

            elif outputCellMode == 123: 
                with open(xyfullPath,'wb') as fp:
                    pickle.dump(self.X,fp)
                    pickle.dump(self.Y1,fp)
                    pickle.dump(self.Y2,fp)
                    pickle.dump(self.Y3,fp)
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
                
            return batchX,batchY
        #------------------------------------
        #------------------------------------
        elif self.outputCellMode == 12 or self.outputCellMode == 23:
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
        
            batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
            batchY1 = self.y1Train[self.batchRandInd[sInd:eInd]]
            batchY2 = self.y2Train[self.batchRandInd[sInd:eInd]]
            
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            
            return batchX,batchY1,batchY2
        #------------------------------------
        #------------------------------------
        elif self.outputCellMode == 123:
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
        
            batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
            batchY1 = self.y1Train[self.batchRandInd[sInd:eInd]]
            batchY2 = self.y2Train[self.batchRandInd[sInd:eInd]] 
            batchY3 = self.y3Train[self.batchRandInd[sInd:eInd]]
            
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            
            return batchX,batchY1,batchY2,batchY3
                
#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    
    isWindows = False

    # Mode 設定
    inputCellmode = int(sys.argv[1])
    dataMode = int(sys.argv[2])
    outputCellMode = int(sys.argv[3])
    datapickleMode = int(sys.argv[4])
    

    # b1単独予測器の場合は-1をつけたpickle指定 
    # b2単独予測器の場合は_2をつけたpickle指定
    # b1b2組み合わせ予測器の場合は_12をつけたpickle指定

    if dataMode == 12:
        
        dataPath = 'b1b2'   
        picklePath = 'xydatab1.pkl'
        trainingpicklePath = 'traintestdatab1.pkl'
        fname = 'kde_fft_log_20*'

    if dataMode == 123:
        dataPath = 'b1b2b3'   
        picklePath = 'xydatab1b2b3.pkl'
        trainingpicklePath = 'traintestdatab1b2b3.pkl'
        fname = 'kde_fft_logb2_20*'

            
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

        
    #Reading load log.txt
    if isWindows:
        files = glob.glob('logsb{}\\log_*.txt'.format(dataMode))
    else:
        #pdb.set_trace() 
        files = glob.glob('./logs/logsb1b2-0.8/log_*.txt')
    
    for fID in np.arange(len(files)):
        print('reading',files[fID])

        if isWindows:
            file = files[fID].split('\\')[1]
        else:
            file = files[fID].split('/')[1]
        
        # 地震プレートモデル用のオブジェクト
        log = EarthQuakePlateModel(dataMode,file,nCell=8,nYear=10000)
        log.loadABLV()
        log.convV2YearlyData()
        
        # KDE
        log.KDE()
        log.FFT(widthWindow=10,eFrq=100)
        log.FFT(widthWindow=10,eFrq=100)

        # 保存
        log.plotV(isPlotShow=False,isYearly=False,prefix='kde_fft_')
    
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
    """
        
    
############################################


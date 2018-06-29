# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:14:14 2018

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
import pywt
import glob

import pandas as pd
import seaborn as sns
from scipy import stats

import math
import scipy.optimize

#########################################
class EarthQuakePlateModel:
    logPath = 'logs'
    dataPath = 'data'
    visualPath = 'visualization'

    #--------------------------
    def __init__(self,logName,nCell=8,nYear=10000):
        self.logName = logName
        self.logFullPath = os.path.join(self.logPath,logName)

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
        plt.close()
        fig, figInds = plt.subplots(nrows=8, sharex=True)
    
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
            if isYearly:
                figInds[figInd].plot(np.arange(self.nYear), self.yV[:,figInd])

        fullPath = os.path.join(self.visualPath,"{}_data.png".format(self.logName))
        plt.savefig(fullPath)
        if isPlotShow:
            fig.show()
            plt.show()            
        
        # pklデータの保存
        fullPath = os.path.join(self.dataPath,"{}{}.pkl".format(prefix,self.logName))
        with open(fullPath,'wb') as fp:
            pickle.dump(self.yV,fp)
            pickle.dump(log.B,fp)
    #--------------------------
    
    #--------------------------
    
#########################################

#########################################
class Data:
    logPath = 'logs'
    dataPath = 'data'
    visualPath = 'visualization'

    #--------------------------
    def __init__(self,fname="yVlog_10*",trainRatio=0.8,nCell=8,sYear=2000, eYear=10000,bInd=0, isTensorflow=False,isWindows=True):
                
        self.nCell = nCell
        
        # pklファイルの一覧
        fullPath = os.path.join(self.dataPath,fname)
        files = glob.glob(fullPath)
        
        # データの領域確保
        self.nData = len(files)
        
        nYear = eYear-sYear
        self.X = np.zeros([self.nData,nCell,nYear])
        self.Y = np.zeros([self.nData])
        # データの読み込み
        for fID in np.arange(self.nData):
            
            if isWindows:
                file = files[fID].split('\\')[1]
            else:
                file = files[fID].split('/')[1]
                
            fullPath = os.path.join(self.dataPath,file)
            
            with open(fullPath,'rb') as fp:
                tmpX = pickle.load(fp)
                tmpY = pickle.load(fp)
            
            # データの切り抜き
            tmpX = tmpX[:,sYear:eYear]
            
            # tensorflow用に変形: isTensorflow = False(training.pyで整形)
            #Y: nDataに対応したlogB1(isTenrsor=False,=tmpYだけやったら、bの値が全部入る)
            if isTensorflow:
                self.X[fID] = np.transpose(tmpX,(1,0))
                self.Y[fID] = tmpY[bInd]
            else:
                self.X[fID] = tmpX
                self.Y[fID] = tmpY[bInd]
                #self.Y = tmpY
    
        # XとYの正規化(Yが小さすぎるため,そのもののbを可視化したいときはYの正規化だけはずす）
        #self.minY = np.min(self.Y)
        #self.maxY = np.max(self.Y)
        #self.Y = (self.Y - self.minY)/(self.maxY-self.minY)
        
        #self.minX = np.min(self.X)
        #self.maxX = np.max(self.X)
        #self.X = (self.X - self.minX)/(self.maxX-self.minX)
        
        self.nTrain = np.floor(self.nData * trainRatio).astype(int)
        self.nTest = self.nData - self.nTrain
        
        # ミニバッチの初期化
        self.batchCnt = 0
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        
    #------------------------------------
    #------------------------------------
    # 年数に対して、連続値として扱う
    def KDE(self,nYear=8000):
        kde = np.zeros([self.nCell,nYear])
        
        v_divid = 10.0
        bw = 0.01
    
        dataflag = False
        for dataInd in np.arange(self.nData):
            
            X_cell_year = self.X[dataInd,:,:]
            
            cellflag = False
            for cellInd in np.arange(self.nCell):
                #X_year = X_cell_year[cellInd,:]
                #　セルごとに10分割するための値
                v_width = X_cell_year[cellInd,:].max() / v_divid
                #　速度ｖから地震が何回起きたか相対的に取得
                eqp_num = np.floor(X_cell_year[cellInd,:] / v_width)
                
                # list(float) -> list(int) -> array
                eqp_tmp = list(map(int,eqp_num))
                eqp_num = np.array(eqp_tmp)
                
                # 年数を任意の回数増やした(0回のデータは消える)
                eqp_v = np.repeat(np.arange(0,nYear),eqp_num)
            
                # bw値　指定
                x_grid = np.arange(0,8000)
                kde_model = stats.gaussian_kde(eqp_v,bw_method=bw)
                kde_model = kde_model(x_grid)
                
                
                kde_model = kde_model[np.newaxis,:]
                
                
                if not cellflag:
                    kde_cell = kde_model
                    cellflag = True
                else:
                    kde_cell = np.concatenate((kde_cell,kde_model),axis=0)
                
                kde = kde_cell[np.newaxis,:,:]
                
            if not dataflag:
                self.kde = kde
                dataflag = True
            else:
                self.kde = np.concatenate((self.kde,kde),axis=0)
        
        #fig,figInds = plt.subplots(nrows=8,sharex=True)
        #for figInd in np.arange(len(figInds)):
            
         #   figInds[figInd].plot(self.kde[0,figInd,:])
        
        #plt.show()
        #pdb.set_trace()
        
        return self.kde #[nData,nCell,nYear]
        
         
    def Wavelet(self,width=30):
        
        nYear = 8000
        #周波数領域を設定する(変更の余地あり)
        widths = np.arange(1,width+1)
        fig,figInds = plt.subplots(nrows=self.nCell,sharex=True)
        wavelet_img = np.zeros([self.nCell,width,nYear])
        dataflag = False
        for dataInd in np.arange(self.nData):
            X_cell_year = self.kde[dataInd,:,:]
            
            # Wavlet
            cellflag = False
            
            for cellInd in np.arange(self.nCell):
                #X_year = X_cell_year[cellInd,:] #[8000,]
                wavelet_img,freqs = pywt.cwt(X_cell_year[cellInd,:],widths,'mexh') #[width,Year]
                #figInds[cellInd].imshow(wavelet_img[cellInd],extent=[1, nYear, widths[-1], widths[0]], cmap='gray',aspect='auto',vmax=abs(wavelet_img[cellInd]).max(), vmin=-abs(wavelet_img[cellInd]).max())
        
                wavelet_img = wavelet_img[np.newaxis,:,:]
                if not cellflag:
                    wavelet_img_cell = wavelet_img
                    cellflag = True
                else:
                    wavelet_img_cell = np.concatenate((wavelet_img_cell,wavelet_img),axis=0)
                
            
        
            wavelet_img_cell = wavelet_img_cell[np.newaxis,:,:,:] 
            if not dataflag:
                self.wavelet_img = wavelet_img_cell
                dataflag = True
            else:
                self.wavelet_img = np.concatenate((self.wavelet_img,wavelet_img_cell),axis=0)
        plt.close()
        fig,figInds = plt.subplots(nrows=8,sharex=True)
        for figInd in np.arange(len(figInds)):
            
            figInds[figInd].imshow(wavelet_img_cell[0,figInd,:,:],extent=[1, nYear, widths[-1], widths[0]], cmap='gray',aspect='auto',vmax=abs(wavelet_img_cell[0,figInd,:,:]).max(), vmin=-abs(wavelet_img_cell[0,figInd,:,:]).max())
        
        plt.show()
            
        
        return self.wavelet_img #[nData,nCell,width,year]
        
            
    
    def FFT(self,slice_size=25,eFrq=250,nYear=8000,nCell=8,trainRatio=0.8,isTensorflow=False):
        
        
        dataflag = False
        fft_amp = np.zeros([self.nData,nCell,nYear])
        for dataInd in np.arange(self.nData):
            # 振幅スペクトル(2次元データのみfftできる)
            fft_amp = np.abs(np.fft.fft(self.kde[dataInd,:,:]))
            
            # 1(Hz)から考える
            fft_amp = fft_amp[:,1:eFrq+1]
            
            fft_amp = fft_amp[np.newaxis,:,:]
            
            
            if not dataflag:
                self.fft_amp = fft_amp
                dataflag = True
            else:
                self.fft_amp = np.concatenate((self.fft_amp,fft_amp),axis=0)
        #pdb.set_trace()

        # ナイキスト周波数を考慮した範囲([]内変更必要) 
        #NyquistInd = int(len(fft_amp[0,0,:])/2)
        
        # plot(Nyquist成分と２５０まで)
        #plt.close()
        #fig,figInds = plt.subplots(nrows=8,sharex=True)
        #for figInd in np.arange(len(figInds)):
            
            #figInds[figInd].plot(fft_amp[0,figInd,:NyquistInd])
         #   figInds[figInd].plot(fft_amp[0,figInd,:250])
        
        #plt.show()
        
        nSlice = int(len(self.fft_amp[0,0,:])/slice_size) 
        
        
        #　FFTをかけたデータをsizeで分割する            
        sliceflag = False
        for cnt in np.arange(nSlice):
            sInd = slice_size * cnt
            eInd = sInd + slice_size
            
            fft_slice = self.fft_amp[:,:,sInd:eInd]            
            fft_slice = fft_slice[:,:,:,np.newaxis]
            
            #各々で平均をとる
            fft_slice_mean = np.mean(fft_slice,axis=2)
            
            
            if not sliceflag:
                self.fft_xMean = fft_slice_mean
                sliceflag = True
            else:
                self.fft_xMean = np.concatenate((self.fft_xMean,fft_slice_mean),axis=2) #[nData,nCell,nSlice]
        
        
        # 学習データとテストデータ数
        self.nTrain = np.floor(self.nData * trainRatio).astype(int)
        self.nTest = self.nData - self.nTrain
        
        # ランダムにインデックスをシャッフル
        self.randInd = np.random.permutation(self.nData)
        
        # 学習データ
        self.xTrain = self.fft_xMean[self.randInd[0:self.nTrain]]
        self.yTrain = self.Y[self.randInd[0:self.nTrain]]
        
       
        # 評価データ
        self.xTest = self.fft_xMean[self.randInd[self.nTrain:]]
        self.yTest = self.Y[self.randInd[self.nTrain:]]
        
       
    
    
    # ミニバッチの取り出し
    def nextBatch(self,batchSize,isTensorflow=True):

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
    
#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    isWindows = False 
    
    #Reading load log.txt
    if isWindows:
        files = glob.glob('logs\\log_10*.txt')
    else:
        files = glob.glob('logs/log_10*.txt')

    
    for fID in np.arange(len(files)):
        print('reading',files[fID])

        if isWindows:
            file = files[fID].split('\\')[1]
        else:
            file = files[fID].split('/')[1]
        
        # 地震プレートモデル用のオブジェクト
        log = EarthQuakePlateModel(file,nCell=8,nYear=10000)
        log.loadABLV()
        log.convV2YearlyData()
        log.plotV(isPlotShow=False,isYearly=False,prefix='yV')
    
    
    #実行する意味がない
    #data = Data(fname="yVlog_10*",trainRatio=0.8,nCell=8,sYear=2000, eYear=10000, bInd=0, isTensorflow=False,isWindows=True)
    #data.KDE(nYear=8000)
    
    #data.Wavelet(width=100)
    #data.FFT(slice_size=25,eFrq=250,nYear=8000,nCell=8,trainRatio=0.8,isTensorflow=False)
    
############################################


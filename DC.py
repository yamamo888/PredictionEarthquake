# -*- coding: utf-8 -*-

import os
import glob
import pickle
import pdb
import time
import shutil

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np
from scipy import stats

import pdb

# ---- params ---- #

nCell = 8

# eq. year in logs     
yrInd = 1
yInd = 1
vInds = [2,3,4,5,6,7,8,9]
simlateCell = 8
# いいんかな？
slip = 0

# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400

nYear = 10000


# ---------------- #

# -----------------------------------------------------------------------------
def loadABLV(logFullPath):
    
    data = open(logFullPath).readlines()
    
    B = np.zeros(nCell)
    
    for i in np.arange(1,nCell+1):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        B[i-1] = tmp[1]
        
    # Vの開始行取得
    isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
    vInd = np.where(isRTOL)[0][0]+1
    
    # Vの値の取得（vInd行から最終行まで）
    flag = False
    for i in np.arange(vInd,len(data)):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        
        if not flag:
            V = tmp
            flag = True
        else:
            V = np.vstack([V,tmp])
            
    return V, B

# -----------------------------------------------------------------------------
def convV2YearlyData(V):
        
    # 初めの観測した年
    sYear = np.floor(V[0,yInd])
    yV = np.zeros([nYear,nCell])
    # 観測データがない年には観測データの１つ前のデータを入れる(累積)
    for year in np.arange(sYear,nYear):
        # 観測データがある場合
        if np.sum(np.floor(V[:,yInd])==year):
            # 観測データがあるときはそのまま代入
            yV[int(year)] = V[np.floor(V[:,yInd])==year,vInds[0]:]
        
        # 観測データがない場合
        else:
            # その1つ前の観測データを入れる
            yV[int(year)] = yV[int(year)-1,:]
    # 累積速度から、速度データにする
    deltaV = yV[yInd:]-yV[:-yInd]
    # 一番最初のデータをappendして、10000年にする
    yV = np.concatenate((yV[np.newaxis,0],deltaV),0)
    # shape=[8000,3]
    yV = np.concatenate((yV[stateYear:,2,np.newaxis],yV[stateYear:,4,np.newaxis],yV[stateYear:,5,np.newaxis]),1)
    
    return yV
    
# -----------------------------------------------------------------------------
def MinErrorNankai(gt,pred):
    
    # ----
    # 真値の地震年数
    gYear = np.where(gt[:,0] > slip)[0]
    # ----

    flag = False
    # Slide each one year 
    for sYear in np.arange(8000-aYear): 
        # 予測した地震の年数 + 1400
        eYear = sYear + aYear

        # 予測した地震年数 
        pYear = np.where(pred[sYear:eYear] > slip)[0]

        # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
        ndist = gauss(gYear,pYear.T)

        # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
        yearError = sum(ndist.max(1)/pYear.shape[0])

        if not flag:
            yearErrors = yearError
            flag = True
        else:
            yearErrors = np.hstack([yearErrors,yearError])
    
    # 最小誤差開始修了年数(1400年)取得
    sInd = np.argmax(yearErrors)
    eInd = sInd + aYear

    # 最小誤差確率　
    maxSim = yearErrors[sInd]
    
    return pred[sInd:eInd,:], maxSim

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss    
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # ---- path ---- #
    # dataPath
    logsPath = 'logs'
    # simulated path
    dataPath = 'b2b3b4b5b6400-450'
    fname = '*.txt' 
    # -------------- #
    
    # --------------------------------------------------------------
    # simulated rireki
    filePath = os.path.join(logsPath,dataPath,fname)
    files = glob.glob(filePath)
    # --------------------------------------------------------------
    
    # --------------------------------------------------------------
    # loading nankai trough (ground truth)
    with open(os.path.join(logsPath,"nankairireki.pkl"),'rb') as fp:
        gtyV = pickle.load(fp)
    # --------------------------------------------------------------
    
    for tfID in [0,1,190]:
        
        # ---- reading gt ---- # 
        # 全領域と確実領域の南海トラフ巨大地震履歴
        gtU = gtyV[tfID,:,:]
        # deltaU -> slip velocity 
        gtUV = np.vstack([np.zeros(3)[np.newaxis], gtU[1:,:] - gtU[:-1,:]])
        # -------------------- #
        
        flag = False
        for fID in np.arange(len(files)):
            
            # file path
            logFullPath = files[fID]
            
            # loading logs 
            V,B = loadABLV(logFullPath)
            
            # V -> yV 
            yV = convV2YearlyData(V)
            
            # maxSim : Degree of Similatery
            # minimum error yV ,shape=[1400,3]
            minyV, maxSim = MinErrorNankai(gtUV,yV)
            
            if not flag:
                maxSims = maxSim
                paths = logFullPath
                flag = True
            else:
                maxSims = np.hstack([maxSims,maxSim])
                paths = np.hstack([paths,logFullPath])
        
        maxSimInds = np.argsort(maxSims)[::-1]
        # path for reading
        maxpaths = paths[maxSimInds]
        # sort degree of similatery 
        maxSim = maxSims[maxSimInds]
        
        # ----------------------------------------------------------
        for line in maxpaths:
            # output csv for path
            with open(f"path_{tfID}_{dataPath}.txt","a") as f:
                f.write(line + "\n")
        for ms in maxSim:
            # output csv for path
            with open(f"DS_{tfID}_{dataPath}.txt","a") as f:
                f.write(str(ms) + "\n")
        # ----------------------------------------------------------
        
        
        
        
        
        
    
    
    
    
    
    




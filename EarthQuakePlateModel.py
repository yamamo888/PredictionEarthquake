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
import matplotlib.pylab as plt
import pywt

#########################################
class EarthQuakePlateModel:
	logPath = 'logs'
	visualPath = 'visualization'
	nCell = 8
	nYear = 10000
	yInd = 1
	vInds = [2,3,4,5,6,7,8,9]
	yV = np.zeros([nYear,nCell])
	
	
	#--------------------------
	def __init__(self,logName):
		self.logName = logName
		self.logFullPath = os.path.join(self.logPath,logName)
	#--------------------------

	#--------------------------
	#データの読み込み
	def loadABLV(self):
		print('reading',self.logFullPath)
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
	#--------------------------
		
	#--------------------------
	# Vの生データのプロット
	def plotV(self,isPlot=False,isYearly=False):
		fig, figInds = plt.subplots(nrows=8, sharex=True)
	
		for figInd in np.arange(len(figInds)):
			figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
			if isYearly:
				figInds[figInd].plot(np.arange(self.nYear), self.yV[:,figInd])

		plt.hold(True)
		
		fullpath = os.path.join(self.visualPath,"{}_data.png".format(self.logName))
		plt.savefig(fullpath)

		if isPlot:
			fig.show()
			plt.show()
	#--------------------------
	
	#--------------------------
	def plotVWavelet(self):
		fig, figInds = plt.subplots(nrows=8, sharex=True)
		
		width = 30
		widths = np.arange(1,width)
		
		for figInd in np.arange(len(figInds)):
			cwtmatr, freqs = pywt.cwt(self.V[:,2+figInd], widths, 'mexh')
			pdb.set_trace()
			
			figInds[figInd].imshow(cwtmatr, extent=[-1, 1, 1, 30], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

		plt.hold(True)
				
		fullpath = os.path.join(self.visualPath,"{}_waveletdata.png".format(self.logName))
		plt.savefig(fullpath)

		fig.show()
		plt.show()
	#--------------------------

#########################################

############## MAIN #####################
if __name__ == "__main__":
	
	#Reading load log.txt
	
	log = EarthQuakePlateModel("log.txt")
	log.loadABLV()
	log.convV2YearlyData()
	log.plotV(isPlot=True,isYearly=True)
	#log.plotVWavelet()
	
	log1 = EarthQuakePlateModel("log_1.txt")
	log1.loadABLV()
	log1.convV2YearlyData()	
	log1.plotV(isPlot=True,isYearly=True)
	#log1.plotVWavelet()
	
	pdb.set_trace()
	
'''
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
'''	
	

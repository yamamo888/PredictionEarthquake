# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import tensorflow as tf
import matplotlib.pylab as plt
import pickle
import pywt
import glob
import EarthQuakePlateModel as eqp
import pdb

nCell = 8
nFreqs = 30
sYear = 2000
eYear = 6000
bInd = 0
myData = eqp.Data(fname="log_10*",trainRatio=0.8,nCell=nCell,nFreqs=nFreqs, sYear=sYear, eYear=eYear, bInd=bInd, isTensorflow=True, isWindows=False)

#myData.meanIng.shape[30,4000,8]
#myData.X.shape[11,30,4000,8]

def plot(img,fname):
	visualPath = 'images'
	nYear = 4000
	widths = np.arange(1,30)
	plt.close()
	fig, figInds = plt.subplots(nrows=8, sharex=True)
	
	
	for figInd in np.arange(len(figInds)):
		#pdb.set_trace()
		figInds[figInd].imshow(img[:,:,figInd], extent=[1, nYear, widths[-1], widths[0]], cmap='gray', aspect='auto',vmax=abs(img[figInd]).max(), vmin=-abs(img[figInd]).min())
			
		
	fullPath = os.path.join(visualPath,"{}_data.png".format(fname))
	plt.savefig(fullPath)


pdb.set_trace()
plot(myData.meanImg,fname='mean')
loglen = 600
for i in np.arange(loglen):
	plot(myData.X[i]-myData.meanImg,fname='{}'.format(i))
#wavelet()

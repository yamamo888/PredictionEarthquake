# -*- co    ding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

import pdb

import makingData as myData
import plot as myPlot

# -------------------------- command argment ----------------------------------
# Model type 0: ordinary regression, 1: anhor-based, 2: atr-nets
methodModel = int(sys.argv[1])
# noize of x1, x2
sigma = np.float(sys.argv[2])
# number of class
nClass = int(sys.argv[3])
# number of rotation -> sin(pNum*pi) & cos(pNum*pi)
pNum = int(sys.argv[4])
# number of layer for Regression NN
depth = int(sys.argv[5])
# -----------------------------------------------------------------------------

# --------------------------- parameters --------------------------------------
dInput = 2
# node of 1 hidden
nHidden = 128
# node of 2 hidden
nHidden2 = 128

# node of output in Regression 
nRegOutput = 1
# node of input in Regression
if methodModel == 0:
    nRegInput = dInput
else:
    nRegInput = nRegOutput + dInput
# node of 1 hidden
nRegHidden = 128
# node of 2 hidden
nRegHidden2 = 128
# node of 3 hidden
nRegHidden3 = 128
# node of 4 hidden
nRegHidden4 = 128

if methodModel == 2:
    isATR = True
else:
    isATR = False
  
# round decimal 
limitdecimal = 3
# maximum of target variables
yMax = 6
# miinimum of target variables
yMin = 2
# Width class
beta = np.round((yMax - yMin) / nClass,limitdecimal)
# Center variable of the first class
first_cls_center = np.round(yMin + (beta / 2),limitdecimal)

# Learning rate
lr = 1e-4
# number of training
nTraining = 2000
# batch size
batchSize = 100
# batch count initializer
batchCnt = 0
# test count
testPeriod = 500
# if plot == True
isPlot = True
# -----------------------------------------------------------------------------

# --------------------------- data --------------------------------------------
# Get train & test data, shape=[number of data, dimention]
x1Train, x2Train, yTrain, x1Test, x2Test, yTest, y = myData.SplitTrainTest(yMin=yMin,yMax=yMax,pNum=pNum,noise=sigma)
# Get anotation y
yTrainlabel, yTestlabel = myData.AnotationY(y,yMin=yMin,yMax=yMax,nClass=nClass,beta=beta)
# x = x1 + x2 shape=[num of data, 2(dim)] 
xTrain = np.concatenate([x1Train,x2Train], 1)
xTest = np.concatenate([x1Test,x2Test], 1)
# -----------------------------------------------------------------------------

#------------------------- placeholder ----------------------------------------
# input of placeholder for classification
x_cls = tf.placeholder(tf.float32,shape=[None,dInput])
# input of placeholder for regression
x_reg = tf.placeholder(tf.float32,shape=[None,nRegInput])
# GT output of placeholder (target)
y = tf.placeholder(tf.float32,shape=[None,1])
# GT output of label
y_label = tf.placeholder(tf.int32,shape=[None,nClass])

# -----------------------------------------------------------------------------

#-----------------------------------------------------------------------------#      
def weight_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
#-----------------------------------------------------------------------------#
def bias_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
#-----------------------------------------------------------------------------#
def alpha_variable(name,shape):
    alphaInit = tf.random_normal_initializer(mean=0.5,stddev=0.1)
    return tf.get_variable(name,shape,initializer=alphaInit)
#-----------------------------------------------------------------------------#
def fc_sigmoid(inputs,w,b,keepProb):
    sigmoid = tf.matmul(inputs,w) + b
    sigmoid = tf.nn.dropout(sigmoid,keepProb)
    sigmoid = tf.nn.sigmoid(sigmoid)
    return sigmoid
#-----------------------------------------------------------------------------#
def fc_relu(inputs,w,b,keepProb):
     relu = tf.matmul(inputs,w) + b
     relu = tf.nn.dropout(relu, keepProb)
     relu = tf.nn.relu(relu)
     return relu
#-----------------------------------------------------------------------------#
def fc(inputs,w,b,keepProb):
     fc = tf.matmul(inputs,w) + b
     fc = tf.nn.dropout(fc, keepProb)
     return fc
#-----------------------------------------------------------------------------#
def Classify(x,reuse=False):
    
    """
    4 layer fully-connected classification networks.
    Activation: relu -> relu -> none
    Dropout: keepProb
    
    Args:
        x: input data (feature vector, shape=[None, number of dimention])
        reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
    
    Returns:
        y: predicted target variables of class (one-hot vector)
    """
    with tf.variable_scope('Classify') as scope:  
        keepProb = 1.0
        if reuse:
            keepProb = 1.0            
            scope.reuse_variables()
        
        # 1st layer
        w1 = weight_variable('w1',[dInput,nHidden])
        bias1 = bias_variable('bias1',[nHidden])
        h1 = fc_relu(x,w1,bias1,keepProb)
        
        # 2nd layer
        w2 = weight_variable('w2',[nHidden,nHidden2])
        bias2 = bias_variable('bias2',[nHidden2])
        h2 = fc_relu(h1,w2,bias2,keepProb) 
        
        # 3rd layar
        w3 = weight_variable('w3',[nHidden2,nClass])
        bias3 = bias_variable('bias3',[nClass])
        y = fc(h2,w3,bias3,keepProb)
        
        # shape=[None,number of class]
        return y
#-----------------------------------------------------------------------------#
def Regress(x_reg,reuse=False,isATR=False,depth=0):
    
    """
    Fully-connected regression networks.
    
    Activation of Atr-nets: relu -> relu -> sigmoid
    Activation of Baseline regression & anchor-based: relu -> relu -> none
    Dropout: keepProb
    
    Args:
        x: input data (feature vector or residual, shape=[None, number of dimention])
        reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
        isR=False : atr-nets, isR=True : ordinary regression & anchor-based (in order to change output activation.)
        depth=3: 3layer, depth=4: 4layer, depth=5: 5layer
    
    Returns:
        y: predicted target variables or residual
    """
    
    with tf.variable_scope("Regress") as scope:  
        keepProb = 1.0
        if reuse:
            keepProb = 1.0            
            scope.reuse_variables()
        
        # 1st layer
        w1_reg = weight_variable('w1_reg',[nRegInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        h1 = fc_relu(x_reg,w1_reg,bias1_reg,keepProb)
        
        if depth == 3:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegOutput])
            bias2_reg = bias_variable('bias2_reg',[nRegOutput])
            
            if isATR:
                # shape=[None,number of dimention (y)]
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb)
            else:
                return fc(h1,w2_reg,bias2_reg,keepProb)
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegOutput])
            bias3_reg = bias_variable('bias3_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb)
            else:
                return fc(h2,w3_reg,bias3_reg,keepProb)
        # ---------------------------------------------------------------------
        elif depth == 5:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer 
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegHidden3])
            bias3_reg = bias_variable('bias3_reg',[nRegHidden3])
            h3 = fc_relu(h2,w3_reg,bias3_reg,keepProb)
            
            # 4th layer
            w4_reg = weight_variable('w4_reg',[nRegHidden3,nRegOutput])
            bias4_reg = bias_variable('bias4_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb)
            else:
                return fc(h3,w4_reg,bias4_reg,keepProb) 
#-----------------------------------------------------------------------------#
def CreateRegInputOutput(x,y,cls_score):
    
    """
    Create input vector(=cls_center_x) & anchor-based method GT output(=r) for Regress.
    
    Args:
        x: feature vector (input data) 
        cls_score: output in Classify (one-hot vector of predicted y class)
    
    Returns:
        pred_cls_center: center variable of class
        r: residual for regression (gt anchor-based) 
        cls_cener_x: center variable of class for regression input
    """
    
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + first_cls_center
    # feature vector + center variable of class
    cls_center_x =  tf.concat((pred_cls_center,x),axis=1)
    # residual = objective - center variavle of class 
    r = y - pred_cls_center
    
    return pred_cls_center, r, cls_center_x
#-----------------------------------------------------------------------------#
def TruncatedResidual(r,reuse=False):
    """
    Truncated range of residual by sigmoid function.
    
    Args:
        r: residual
        reuse=False: Train, reuse=True: Evaluation & Test (alpha sharing)
    
    Returns:
        r_at: trauncated range of residual
        alpha: traincated adjustment parameter
    """
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        
        alpha = alpha_variable("alpha",[1]) 
        
        r_at = 1/(1 + tf.exp(- alpha * r))
        
        return r_at, alpha
#-----------------------------------------------------------------------------#
def Reduce(r_at,param,reuse=False):
    """
    Reduce truncated residual(=r_at) to residual(=pred_r).
    
    Args:
        r_at: truncated residual
        param: truncated adjustment param (alpha)
        reuse=False: Train, reuse=True: Evaluation & Test (alpha sharing)
    
    Returns:
        pred_r: reduce residual 
    """
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        
        pred_r = (-1/r_at) * tf.log((1/param) - 1)
        
        return pred_r
#-----------------------------------------------------------------------------#    
def Loss(y,predict,isCE=False):
    """
    Loss function for Regress & Classify & alpha.
    Regress & alpha -> Mean Absolute Loss(MAE), Classify -> Cross Entropy(CE)
    
    Args:
        y: ground truth
        predict: predicted y
        isR=False: CE, isR=True: MAE
    """
    if isCE:
        return tf.losses.softmax_cross_entropy(y,predict)
    else:
        return tf.reduce_mean(tf.abs(y - predict))
#-----------------------------------------------------------------------------#
def LossGroup(self,weight1): 
    group_weight = tf.reduce_sum(tf.square(weight1),axis=0)
    return group_weight
#-----------------------------------------------------------------------------#
def Optimizer(loss,name_scope="Regress"):
    """
    Optimizer for Regress & Classify & alpha.
    
    Args:
        loss: loss function
        name_scope: "Regress" or "Classify" or "TrResidual"
    """
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
    opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=Vars)
    return opt
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
def main():
    
    # =================== Classification networks =========================== #
    # IN -> x_cls: feature vector x = x1 + x2
    # OUT -> cls_op: one-hot vector
    cls_op = Classify(x_cls)
    cls_op_test = Classify(x_cls,reuse=True)
    
    # ============== Make Regression Input & Output ========================= #
    # IN -> x_cls: feature vector x = x1 + x2, cls_op: one-hot vector
    # OUT -> pred_cls_center: center of predicted class, res: gt residual, reg_in: x + pred_cls_center
    pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,y,cls_op)
    pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,y,cls_op_test)
    
    # ====================== Regression networks ============================ #
    # IN -> x_reg: feature vector x = x1 + x2 (only baseline) or x + predicted center of class, 
    #       isATR: bool (if ATR-Nets, isATR=True), depth: number of layer (command args)
    # OUT -> reg_op: predicted of target variables y (baseline), predicted residual (Anchor-based), predicted truncated residual (ATR-Nets)
    reg_op = Regress(x_reg,isATR=isATR,depth=depth)
    reg_op_test = Regress(x_reg,reuse=True,isATR=isATR,depth=depth)
    
    # =================== Truncated residual ================================ #
    # IN -> res: residual, [None,1]
    # OUT -> res_at: truncated range residual, [None,1], alpha_op: truncated parameter, [1]  
    res_atr, alpha_op = TruncatedResidual(res)
    res_atr_test, alpha_op_test = TruncatedResidual(res_test,reuse=True)
    
    # ================== Reduce truncated residual ========================== #
    # IN -> reg_op: predicted truncated regression
    # OUT -> reduce_res: reduced residual, [None,1] 
    reduce_res_op = Reduce(reg_op,alpha_op,reuse=True)
    reduce_res_op_test = Reduce(reg_op_test,alpha_op_test,reuse=True)
    
    # predicted y by ATR-Nets
    pred_y = pred_cls_center + reduce_res_op
    pred_y_test = pred_cls_center_test + reduce_res_op_test
    
    # ============================= Loss ==================================== #
    # Classification loss
    # gt label (y_label) vs predicted label (cls_op)
    loss_cls = Loss(y_label,cls_op,isCE=True)
    loss_cls_test = Loss(y_label,cls_op_test,isCE=True)
    
    # Baseline regression loss train & test
    # gt value (y) vs predicted value (reg_op)
    loss_reg = Loss(y,reg_op)
    loss_reg_test = Loss(y,reg_op_test)
    
    # Regression loss for Anchor-based
    # gt residual (res) vs predicted residual (res_op)
    loss_anc = Loss(res,reg_op)
    loss_anc_test = Loss(res_test,reg_op_test)
    
    # Regression loss for Atr-nets
    # gt truncated residual (res_at) vs predicted truncated residual (res_op)
    loss_atr = Loss(res_atr,reg_op)
    loss_atr_test = Loss(res_atr_test,reg_op_test)
    
    # Training alpha loss
    # gt value (y) vs predicted value (pred_yz = pred_cls_center + reduce_res)
    loss_alpha = Loss(y,pred_y)
    loss_alpha_test = Loss(y,pred_y_test)
    
    # ========================== Optimizer ================================== #
    # for classification 
    trainer_cls = Optimizer(loss_cls,name_scope="Classify")
    
    # for Baseline regression
    trainer_reg = Optimizer(loss_reg)
    
    # for Anchor-based regression
    trainer_anc = Optimizer(loss_anc)
    
    # for Atr-nets regression
    trainer_atr = Optimizer(loss_atr)
    
    # for alpha training in atr-nets
    trainer_alpha = Optimizer(loss_alpha,name_scope="TrResidual")
    
    #------------------------------------------------------------------------ # 
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # save model
    saver = tf.train.Saver()
    #------------------------------------------------------------------------ #
    #------------------------------------------------------------------------ #
    
    # start training
    flag = False
    for i in range(nTraining):
        
        # Get mini-batch
        batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel,batchSize,batchCnt)
        
        # ==================== Baseline regression ========================== #
        if methodModel == 0:
            # regression
            _, trainPred, trainRegLoss = sess.run([trainer_reg, reg_op, loss_reg], feed_dict={x_reg:batchX, y:batchY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:   
                # regression
                testPred, testRegLoss = sess.run([reg_op_test, loss_reg_test], feed_dict={x_reg:xTest, y:yTest})
            
                trainTotalVar  = np.var(np.abs(batchY - trainPred))
                testTotalVar = np.var(np.abs(yTest - testPred))
                
                print("itr:%d, trainRegLoss:%f, trainTotalVar:%f" % (i, trainRegLoss, trainTotalVar))
                print("itr:%d, testRegLoss:%f, testTotalVar:%f" % (i, testRegLoss, testTotalVar)) 
                
                
                if not flag:
                    trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
            
        # ==================== Anchor-based regression ====================== #
        elif methodModel == 1:
            # classication
            _, trainClsCenter, trainClsLoss = sess.run([trainer_cls, pred_cls_center, loss_cls], feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            # feature vector in regression
            trInReg = np.concatenate([trainClsCenter,batchX],1)            
            # regression
            _, trainResPred, trainResLoss = sess.run([trainer_anc, reg_op, loss_anc],feed_dict={x_cls:batchX, x_reg:trInReg ,y:batchY, y_label:batchlabelY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                # classication
                testClsLoss, testClsCenter = sess.run([loss_cls_test, pred_cls_center_test], feed_dict={x_cls:xTest, y:yTest, y_label:yTestlabel})    
                # feature vector in regression
                teInReg = np.concatenate([testClsCenter,xTest],1)
                # regression
                testResLoss, testResPred = sess.run([loss_anc_test, reg_op_test], feed_dict={x_cls:xTest, x_reg:teInReg ,y:yTest, y_label:yTestlabel})
                
                # Reduce
                trainPred = trainClsCenter + trainResPred
                testPred = testClsCenter + testResPred     
                
                # total loss (mean) & variance
                trainTotalLoss = np.mean(np.abs(batchY - trainPred))
                trainTotalVar  = np.var(np.abs(batchY - trainPred))
                testTotalLoss  = np.mean(np.abs(yTest - testPred))
                testTotalVar  = np.var(np.abs(yTest - testPred))
                
                print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
                print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
                
                if not flag:
                    trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                    trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                    trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                    flag = True
                else:
                    trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                    trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                    trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
            
        # ======================== Atr-Nets ================================= #
        elif methodModel == 2:
            # classication
            _, trainClsCenter, trainClsLoss = sess.run([trainer_cls, pred_cls_center, loss_cls], feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            # feature vector in regression
            trInReg = np.concatenate([trainClsCenter,batchX],1)            
            # regression
            _, trainResPred, trainResLoss = sess.run([trainer_atr, reg_op, loss_atr], feed_dict={x_cls:batchX, x_reg:trInReg, y:batchY, y_label:batchlabelY})
            # alpha
            _, trainAlpha, trainRResPred, trainAlphaLoss = sess.run([trainer_alpha, alpha_op, reduce_res_op, loss_alpha], feed_dict={x_cls:batchX, x_reg:trInReg, y:batchY, y_label:batchlabelY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                # classication
                testClsLoss, testClsCenter = sess.run([loss_cls_test, pred_cls_center_test], feed_dict={x_cls:xTest, y:yTest, y_label:yTestlabel})    
                # feature vector in regression
                teInReg = np.concatenate([testClsCenter,xTest],1)
                # regression
                testResLoss, testResPred = sess.run([loss_atr_test, reg_op_test], feed_dict={x_cls:xTest, x_reg:teInReg, y:yTest, y_label:yTestlabel})
                # test alpha
                testAlphaLoss, testAlpha, testRResPred = sess.run([loss_alpha_test, alpha_op_test, reduce_res_op_test], feed_dict={x_cls:xTest, x_reg:teInReg, y:yTest, y_label:yTestlabel})
                
                # Recover
                trainPred = trainClsCenter + trainRResPred
                testPred = testClsCenter + testRResPred
            
                # total loss (mean) & variance
                trainTotalLoss = np.mean(np.abs(batchY - trainPred))
                trainTotalVar  = np.var(np.abs(batchY - trainPred))
                testTotalLoss  = np.mean(np.abs(yTest - testPred))
                testTotalVar  = np.var(np.abs(yTest - testPred))
                
                TrainResidualat = 1/(1+np.exp(-trainAlpha * trainResPred))
                TrainBigResidual = np.where((0.0==TrainResidualat)|(TrainResidualat==1.0))
                bigResidualpar = TrainBigResidual[0].shape[0] / batchY.shape[0]
                
                TestTrRes = 1/(1+np.exp(-testAlpha * testResPred))
                TestBigResidual = np.where((0.0==TestTrRes)|(TestTrRes==1.0))
                TestbigResidualpar = TestBigResidual[0].shape[0] / yTest.shape[0]
                
                print("Test Alpha", testAlpha)
                print("BigTrainResidual割合", bigResidualpar)
                print("BigTestResidual割合", TestbigResidualpar)
                print("-----------------------------------")
                print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
                print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
                
                if not flag:
                    trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                    trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                    trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                    flag = True
                else:
                    trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                    trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                    trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
    
    # ------------------------- save model  --------------------------------- #
    """
    modelFileName = "model_{}_{}_{}_{}_{}_{}.ckpt".format(methodModel,sigma,nClass,pNum,nTrain,nTest)
    modelPath = "models"
    modelfullPath = os.path.join(modelPath,modelFileName)
    saver.save(sess,modelfullPath)
    """
    # ------------------------- plot loss & toydata ------------------------- #
    if methodModel == 0:
        myPlot.Plot_loss(0, 0, 0, 0, trainRegLosses, testRegLosses, testPeriod,isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth)
        myPlot.Plot_3D(xTest[:,0],xTest[:,1],yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth, isTrain=0)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth, isTrain=1)
        
    elif methodModel == 1:
        myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, testPeriod, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth)   
        myPlot.Plot_3D(xTest[:,0],xTest[:,1],yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth, isTrain=0)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth, isTrain=1)
        
    elif methodModel == 2: 
        myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, testPeriod,isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=testAlpha, pNum=pNum, depth=depth)    
        myPlot.Plot_3D(xTest[:,0],xTest[:,1],yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=testAlpha, pNum=pNum, depth=depth, isTrain=0)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot,methodModel=methodModel,sigma=sigma,nClass=nClass,alpha=trainAlpha,pNum=pNum,depth=depth,isTrain=1)
        
    # ----------------------------------------------------------------------- #  
    
if __name__ == "__main__":
    main()
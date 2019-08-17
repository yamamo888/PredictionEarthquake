# Tensorflow ��p���� Baseline Regression ��Anchor-based Regression �ƒ�Ė@ ATR-Nets �̎���

���� `README.md` �ɂ́A�e�R�[�h�̎��s���ʁA�e�R�[�h�̐������L�ڂ��Ă��܂��B
���s�t�@�C���� `trainingModel.py`�ŁA �f�[�^�쐬�� `makingData.py`�A ���ʂ̉摜�쐬�E�o�͂� `plot.py` �ŁA�s���Ă��܂��B`makingData.py` �� `plot.py` �́A���s�t�@�C������Ăяo����܂��B


## ���� [Contents]

0. [�g����](#ID_0)
	1. [�R�}���h](#ID_0-1)

1. [�g�p����点��K�i�f�[�^ : `makingData.py`](#ID_1)
	1. [�R�[�h�̐���](#ID_1-1)
	2. [�点��K�i�̗�(�R�[�h�̎��s����)](#ID_1-2)

2. [�e��@��Graph�쐬 (tensorflow��) : `trainingMdel.py`](#ID_2)
	1. [�p�����[�^](#ID_2-1-1)
	2. [����NN�Ɖ�ANN](#ID_2-1-2)
	3. [Anchor-based regression��ATR-Nets�̉�ANN�Ŏg�p������͂Əo�͂̍쐬](#ID_2-1-3)
	4. [ATR-Nets�̍H�v�_](#ID_2-1-4)
	5. [�덷�֐��E�œK��](#ID_2-1-5)
	6. [�֐��̌Ăяo��](#ID_2-1-6)

3. [�e��@��Graph���s (python��) : `trainingMdel.py`](#ID_3)
	1. [�~�j�o�b�`(�w�K�f�[�^) : `makingData.py`](#ID_3-1)
	2. [Baseline Regression](#ID_3-2)
	3. [Anchor-based regression](#ID_3-3)
	4. [ATR-Nets](#ID_3-4)
	5. [���f���̕ۑ�](#ID_3-5)

3. [���s���� : `plot.py`](#ID_4)

<a id="ID_0"></a>

## �g����

<a id="ID_0-1"></a>

### �R�}���h

```
python trainingModel.py <���f���̎��(methodModel)> <�m�C�Y(sigma)> <�N���X��(number of class)> <��]��(number of rotation)> <�K�w��(number of layer in Regression NN)>
```
- ��F���f���� Anchor-based Regression�A�����ϐ��̕��U�� 0.00001�A�N���X���� 10�A��]�� 5�A3�K�w��ANN���g�p�������ꍇ :
```python trainingModel.py 1 0.00001 10 5 3```

- �N���X���� Baseline Regression�̎��ɂ͕K�v�Ȃ����A�w�肷��K�v����


<br>

### �R�[�h�̐���
- ���f���̎�ސݒ� `methodModel` �� 0 �̂Ƃ� Baseline Regression�A1 �̂Ƃ� Anchor-based Regression�A2 �̂Ƃ� ATR-Nets �����s����
- �����ϐ��̕��U `sigma` �� 0.0000001 �ȉ����������� ($$x_1$$,$$x_2$$ �̑傫�����������̂�)
- �ړI�ϐ��̃N���X�� `nClass` �� 10,20,50����������
- �����ϐ��̉�]�� `pNum`�� 2��3��5���炢���������� (1���ƕs���肪�N����Ȃ��A5�ȏ�͕s���肪�N���肷���邽��)
- ��ANN�̑w�� `depth`�� 3,4,5

<br>

- �R�[�h��̕ϐ��ƃR�}���h����

```python:trainingModel.py
# -------------------------- command arugment ----------------------------------
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
```


<a id="ID_1"></a>

## �g�p����点��K�i�f�[�^ : `makingData.py`
(x1,x2,y)����Ȃ�R�����̂点��K�i�f�[�^���쐬����B
y �� 0 ~ Sigma  �̈�l�������z U(0,Sigma) �ɏ]���Ĕ����������f�[�^�B�ȉ��A�ړI�ϐ�y�Ɛ����ϐ� x1,x2 �̊֌W��:<br>

> ![makedata](/results/makedata.png)


<br>

<a id="ID_1-1"></a>

### �R�[�h�̐���

- �点��K�i�f�[�^�쐬���s���B�쐬�����f�[�^���w�K�p�f�[�^�ƃe�X�g�p�f�[�^�ɕ�������B���������́A`trainRatio`�Ŏw�肵�A�w�K�p�f�[�^80%�A�e�X�g�p�f�[�^20%

- �S�f�[�^�� `nData`�A�w�K�ƃe�X�g�̕������� `trainRatio` �����߂�ƁA�w�K�f�[�^�� `nTrain` �ƃe�X�g�f�[�^�� `nTest` �����܂�B�Ⴆ�΁A`nData = 8000`�̂Ƃ��A`trainRatio = 0.8`�Ƃ���ƁA�w�K�f�[�^�� 6400, �e�X�g�f�[�^�� 1600�ƂȂ�B


```python:makingData.py
# training rate
trainRatio = 0.8
# number of data
nData = 8000
# number of train data
nTrain = int(nData * trainRatio)
# number of test data
nTest = int(nData - nTrain)
# batch random index
batchRandInd = np.random.permutation(nTrain)
```

<br>

```python:makingData.py
def SplitTrainTest(yMin=2,yMax=6,pNum=5,noise=0):
    ...
    
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

    # shape=[number of data, dimention]
    return x1Train, x2Train, yTrain, x1Test, x2Test, yTest, y
```

<br>

- �����̐���
	- x1,x2 �̉�]�� `pNum` �� x1�Ax2�̕��U `noise` �́A `trainingModel.py` �����s����Ƃ��ɃR�}���h�����Ŏw�肳�ꂽ���̂��n�����B
	- �ړI�ϐ��͈̔͂̍ŏ��l `yMin` �ƍő�l `yMax` �́A2,6�B


<br>

- ���ɁA�ړI�ϐ��̃��x���t��������B���ރj���[�����l�b�g���[�N�̂��� Anchor-based Regression �� ATR-Nets�̎��ɕK�v�B


```python:makingData.py
def AnotationY(yMin=2,yMax=6,yClass=10,nClass=10,beta=1):
    ...

    
    
    flag = False
    for nInd in np.arange(target.shape[0]):
        tmpY = target[nInd]
        oneHot = np.zeros(len(yClass))
        ind = 0
        # (�ŏ��A�ő�]
        for threY in yClass:
            if (tmpY > threY) & (tmpY <= threY + beta):
                      oneHot[ind] = 1            
            ind += 1
        # �ŏ��l��0�Ԗڂ̃N���X�ɂ���
        if target[nInd] == yMin:
            oneHot[0] = 1
        # �ő�l����ԍŌ�̃N���X�Ƀ��x�������̂�߂�
        if target[nInd] == yMax:
            oneHot[-2] = 1
        
        tmpY = oneHot[np.newaxis] 
              
        if not flag:
            Ylabel = tmpY
            flag = True
        else:
            Ylabel = np.vstack([Ylabel,tmpY])
            
    # �l�������Ă��Ȃ��N���X���폜
    if len(yClass) == nClass + 1:
        Ylabel = Ylabel[:,:-1]
    
    YTrainlabel = Ylabel[:nTrain]
    YTestlabel = Ylabel[nTrain:]
    
    # shape=[number of data, number of class]
    return YTrainlabel, YTestlabel
```
<br>

- �f�[�^�쐬���̃p�����[�^�̐ݒ�
	- �N���X�� `nClass` �́A `trainingModel.py` �����s����Ƃ��ɃR�}���h�����Ŏw�肷��B�N���X������ `beta` �� `trainingModel.py` �Ōv�Z���ꂽ���̂��n�����B

<br>

<a id="ID_1-2"></a>

- �点��K�i�f�[�^ (�S�f�[�^)
![toydata](/results/toydata.png)

<br>

<a id="ID_1-1"></a>

## �e��@�ɂ��\������: `trainingModel.py`

Baseline Regression �́A��A�j���[�����l�b�g���[�N (NN)�ł���AAnchor-based Regression �́A����NN�Ɖ�ANN��g�ݍ��킹�����̂ł���AATR-Nets�͕���NN�Ɖ�ANN�ɁA�c�����g�傷��l�b�g���[�N��ǉ��������̂ł���B3�̎�@�� `trainingModel.py` �ɂ�1�ɂ܂Ƃ߂Ă���B3�����Ɏ��s���邱�Ƃ͂ł����A�R�}���h������`methodModel`���w�肳�ꂽ���f����1���������s�����B���s���@�̓R�}���h���Q�ƁB <a id="ID_0-1"></a>


- Anchor-based�̃��f���}

![anchor-based](/results/anchor-based.png)

<br>

- ATR-Nets�̃��f���}

![atr-nets](/results/atr-nets.png)


<br>

<a id="ID_2-1-1"></a>

### �p�����[�^

- ����NN�Ɖ�ANN�̊e��m�[�h�̎�����


```python:trainingModel.py
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
```

- �ړI�ϐ��͈̔� U(yMin,yMax)�ƃN���X������ `beta` �Ɓ@���߂̃N���X�̒��S�l `first_cls_center`

```python:trainingModel.py
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
```

- �w�K�� `lr` �ƃo�b�`�T�C�Y `batchSize` �ƃo�b�`�̏����� `batchCnt` (makingData.py�p)
- ATR-Nets�̎� `methodModel == 2` �� `isATR = True`�@����ȊO�́A`isATR = False`


```python:trainingModel.py
# Learning rate
lr = 1e-4
# number of training
nTraining = 500
# batch size
batchSize = 100
# batch count initializer
batchCnt = 0
# test count
testPeriod = 500
# if plot == True
isPlot = True

if methodModel == 2:
    isATR = True
else:
    isATR = False

```

<br>

- `makingData.py`��������ϐ��ƖړI�ϐ�(�w�K�ƃe�X�g�̗���)�Ƃ��󂯎��A�����ϐ��� x �� $$x_1$$,$$x_2$$ �� concat ����2�����̃f�[�^�ɂ���B

```python:trainingData.py
# --------------------------- data --------------------------------------------
# Get train & test data, shape=[number of data, dimention]
x1Train, x2Train, yTrain, x1Test, x2Test, yTest, y = myData.SplitTrainTest()
# Get anotation y
yTrainlabel, yTestlabel, yMin, yMax = myData.AnotationY()
# x = x1 + x2 shape=[num of data, 2(dim)] 
xTrain = np.concatenate([x1Train,x2Train], 1)
xTest = np.concatenate([x1Test,x2Test], 1)
# -----------------------------------------------------------------------------
``` 

<br>

- �d��(`weight_variable`)�ƃo�C�A�X(`bias_variable`)�Asigmoid�֐���alpha�ϐ�(`alpha_variable`)���`����B`alpha` �̏����l `alphaInit` �͕��� `mean` �ƕ��U `stddev` ���w�肷��K�v����B

```python:trainingModel.py
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
```

<br>

- ����
	- name: �ϐ��̖��O�A�Ⴆ�� w_1,b_1,alpha
	- shape: �ϐ��̌`�A�Ⴆ�� [128,128],[128],[1]

<br>

- �������֐��ƑS�������`����B�S���� + sigmoid `fc_sigmoid`�A�S���� + relu `fc_relu`�A�S�����̂� `fc`���`�B
- ���͂́A���� inputs�A�d�� w�A�o�C�A�X b�A�h���b�v�A�E�g�� keepProb ���w��B
- keepProb�̓h���b�v�A�E�g����m�[�h�����w��ł���B�Ⴆ�΁A0.5 �̏ꍇ�͔����̃m�[�h�����g�p����Ȃ��B�������A�e�X�g����1.0�ɂ��Ȃ���΂Ȃ�Ȃ��B


```python:trainingModel.py

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
```

<br>

<a id="ID_2-1-2"></a>

### ����NN�Ɖ�ANN

- ����NN

	- {���͑w: 2�m�[�h�A�B��w1: 128�m�[�h�A�B��w2: 128�m�[�h�A�o�͑w: 1�m�[�h} �̕���NN (`Classify`) �́AAnchor-based Regression��ATR-Nets�Ŏg�p�B
	- ���� x ��2�����s��ŁA�o�� y �̓N���X������1�����x�N�g�� (0 ~ 1�̃N���X�m���ɑΉ�)�B
	- �e��m�[�h���̓p�����[�^�̍��ڂ��Q�Ɓ@



```python:trainingModel.py
def Classify(x,reuse=False):
    ...
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

```
- ����
	- �e�X�g���� reuse=True �ɂ��āA�d�݂ƃo�C�A�X�����L����B	

<br>

- ��ANN
	- ��ANN (`Regress`) �͂��ׂĂ̎�@�ŗp����B<font color="Red">�������AATR-Nets�̎������o�͑w�̊������֐���sigmoid�Ɏw��(exp�����Ƃ��Ƀ}�C�i�X�l�͌v�Z�ł��Ȃ�����)�B

	- {���͑w: 2�m�[�h or 3�m�[�h�A�B��w1: 128�m�[�h�A�B��w2: 128�m�[�h�A�o�͑w: 1�m�[�h} �̉�ANN (`Regress`)

	- ���� `x_reg` �́ABaseline Regression�̎���2�����s��̐����ϐ��ŁAAnchor-based regression��ATR-Nets�̎���2�����s��̐����ϐ���1�����̃N���X�̒��S�l�Ƃ�concat����3�����s��ł���B

	- �o�͂́ABaseline Regression �̎���1�����x�N�g���̖ړI�ϐ��̗\���l�ŁAAnchor-based regression��ATR-Nets�̎���1�����x�N�g���̎c��(=�^�l�ƃN���X�̒��S�l)�ł���B

	- �e��m�[�h���̓p�����[�^�̍��ڂ��Q�Ɓ@


```python:trainingModel.py
def Regress(x_reg,reuse=False,isATR=False,depth=0):
    ...
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
```
<br>

- ����
	- isATR: ATR-Nets�̎��� True�A���̑��̎�@�� False
	- depth: �K�w���w��@�Ⴆ�� depth=3 �̂Ƃ��� 3 �K�w���f�� ���݂� 3,4,5 �K�w���f���ɑΉ�
	- �e�X�g���� reuse=True �ɂ��āA�d�݂ƃo�C�A�X�����L����B

<br>



<a id="ID_2-1-3"></a>


### Anchor-based regression��ATR-Nets�̉�ANN�Ŏg�p������͂Əo�͂̍쐬

- ��ANN�̓���
	- �N���X�m���̃x�N�g���̍ő�m���̃N���X `pred_maxcls` ����A���̃N���X�̒��S�l `pred_cls_center` ���擾�B�����ϐ���concat����B
- ��ANN�̏o�͂̐^�l
	- �c�� (�ړI�ϐ� - �N���X�̒��S�l `pred_cls_center` )

```python:trainingModel.py
def CreateRegInputOutput(x,y,cls_score):
    ...
    
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + first_cls_center
    # regression input = feature vector + center variable of class
    cls_center_x =  tf.concat((pred_cls_center,x),axis=1)
    # residual = objective - center variavle of class 
    r = y - pred_cls_center
    
    return pred_cls_center, r, cls_center_x
#-----------------------------------------------------------------------------#
```

- ����
	- x: �����ϐ� x = x1 + x2
	- y: �ړI�ϐ�
	- cls_score: �N���X�m�� (`Classify` �̏o��)


<br>

<a id="ID_2-1-4"></a>

### ATR-Nets�̍H�v�_

- �c���͈̔͂�_sigmoid�֐�_��p���āA[0,1] �ɃG���R�[�h����
- sigmoid�֐��̌X�� `alpha` �͊w�K���čœK������
- �c���ƃG���R�[�h���ꂽ�c���Ƃ̊֌W���F <bf>
> ![rat](/results/rat.png)


```python:trainingModel.py
def TruncatedResidual(r,reuse=False):
    ...
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        # traincated adjustment parameter
        alpha = alpha_variable("alpha",[1]) 
        # trauncated range of residual
        r_at = 1/(1 + tf.exp(- alpha * r))
        
        return r_at, alpha
#-----------------------------------------------------------------------------#
```


- ����
	- r: �^�̎c��

<br>

- �c���ƃG���R�[�h���ꂽ�c���̊֌W�} 
![atr](/results/atr.png)

<br>

- �G���R�[�h���ꂽ�c�������Ƃ͈̔͂̎c���ɖ߂� 
- ���F<bf>
> ![r](/results/r.png)

```python:trainingModel.py
def Reduce(r_at,alpha,reuse=False):
    ...
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        # reduce residual
        pred_r = (-1/r_at) * tf.log((1/alpha) - 1)
        
        return pred_r
#-----------------------------------------------------------------------------# 

```

- ����
	- r_at: �G���R�[�h���ꂽ�c��
	- alpha: sigmoid�֐��̌X��

<br>

<a id="ID_2-1-5"></a>

### �����֐��E�œK��

- ����NN�̓N���X�G���g���s�[�덷�ŁA��ANN��alpha���w�K����Ƃ��͐�Ε��ό덷
- _1�̊w�K����w�K����Ƃ��͑��̊w�K��� frozen����B���̂��߁Aname_scope�Ŋw�K����w�K����w�肷��K�v����B�Ⴆ�΁A����NN���w�K����Ƃ��́A���̉�ANN�w�K���alpha�w�K����w�K���Ȃ� (`name_scope="Regress"` ���w��)_


```python:trainingModel.py
def Loss(y,predict,isCE=False):
    ...
    if isCE:
        return tf.losses.softmax_cross_entropy(y,predict)
    else:
        return tf.reduce_mean(tf.abs(y - predict))
#-----------------------------------------------------------------------------#
def Optimizer(loss,name_scope="Regress"):
    ...
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
    opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=Vars)
    return opt

```

- ����
	- isCE: ��ANN��alpha�w�K���鎞�� False�A����NN�̎��� True (default)
	- name_scope: ��ANN�̎��� Regress (default)�A����NN�̎��� Classify�Aalpha�w�K����Ƃ��� TrResidual ���w��@

<br>


<a id="ID_2-1-6"></a>

### �֐��̌Ăяo��

- ����NN

```python:trainingModel.py
    cls_op = Classify(x_cls)
    cls_op_test = Classify(x_cls,reuse=True)
```

- Anchor-based regression��ATR-Nets�̉�ANN�ł́A�����ϐ� x �ƃN���X�̒��S�l `pred_cls_center` �� concat����`reg_in` ����͂Ƃ��A�c�� (�ړI�ϐ� - �N���X�̒��S�l ) `res` ���o�͂̐^�l�ɂ���B


```python:trainingModel.py
    pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,cls_op)
    pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,cls_op_test)
```

- ��ANN
	- Baseline regression �̓��͂͐����ϐ� `x_reg`�AAnchor-based regression��ATR-Nets�̓��͂͐����ϐ� x �ƃN���X�̒��S�l `reg_in`���� (sess.run��)
	- Baseline regression��Anchor-based regression�̎� `isATR`�� False�AATR-Nets�̎� True (�o�͑w�̊������֐���sigmoid�ɂ��邽��)

```python:trainingModel.py
reg_op = Regress(x_reg,isATR=isATR,depth=depth)
reg_op_test = Regress(x_reg,reuse=True,isATR=isATR,depth=depth)
    
```    

- Adaptive Truncated residual 
- ���͂͐^�̎c�� `res`�A�o�͂͐^�̊g�傳�ꂽ�c�� `res_atr` ��sigmoid�֐��̌X�� `alpha_op`


```python:trainingModel.py
res_atr, alpha_op = TruncatedResidual(res)
res_atr_test, alpha_op_test = TruncatedResidual(res_test,reuse=True)
```
- Reduce residual
- ���͂͗\�������g��c�� `res_op`��sigmoid�֐��̌X�� `alpha_op`�A�o�͂͌��ɖ߂����c�� `reduce_res`
- TruncatedResidual()���Œ�`���ꂽalpha���g�p����̂ŁA�w�K�E�e�X�g�����Ƃ� `reuse=True`

```python:trainingModel.py
reduce_res = Reduce(reg_op,alpha_op,reuse=True)
reduce_res_test = Reduce(reg_op_test,alpha_op_test,reuse=True)
```
<br>

- �\�������ړI�ϐ� `pred_y` = �N���X�̒��S�l `pred_cls_center` + ���ɖ߂����c�� `reduce_res_op`


```python:trainingModel.py
# predicted y by ATR-Nets
pred_y = pred_cls_center + reduce_res_op
pred_y_test = pred_cls_center_test + reduce_res_op_test
```
<br>

- �덷�֐�
	- Baseline Regression�� `loss_reg`�AAnchor-based regression�� `loss_cls`��`loss_anc`�AATR-Nets��`loss_cls`��`loss_atr`��`loss_alpha`
	- `loss_cls`�� `isCE=True`
	- `loss_cls`�̓N���X�̃��x���A `loss_reg`�͖ړI�ϐ��A`loss_anc`�͎c���A`loss_atr`�̓G���R�[�h���ꂽ�c�����^�l�ł���B`loss_cls`�͕���NN�̏o�� `cls_op`��`loss_reg`�A`loss_anc`�A`loss_atr`�͉�ANN�̏o�� `reg_op` ���g�p����B


```python:trainingModel.py
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
``` 

- �œK��
	- Baseline Regression�� `trainer_reg`�AAnchor-based regression��`trainer_cls`��`trainer_anc`�AATR-Nets��`trainer_cls`��`trainer_atr`��`trainer_alpha` 
	- name_scope�ŕ���NN��`Classify`�Aalpha�w�K���`TrResidual`���w��B(��ANN�͎w�肷��K�v�Ȃ�)

```python:trainingModel.py
# for classification 
trainer_cls = Optimizer(loss_cls,"Classify")
    
# for Baseline regression
trainer_reg = Optimizer(loss_reg)
    
# for Anchor-based regression
trainer_anc = Optimizer(loss_anc)
    
# for Atr-nets regression
trainer_atr = Optimizer(loss_atr)
    
# for alpha training in atr-nets
trainer_alpha = Optimizer(loss_alpha,name_scope="TrResidual")
```
<br>

<a id="ID_3"></a>


## �e��@��Graph���s (python��) : `trainingMdel.py`
�~�j�o�b�`�f�[�^�擾�A�w�K�t�F�[�Y���s�A�e�X�g�t�F�[�Y���s��3�̒i�K�ɑ傫����������B

<br>



<a id="ID_3-1"></a>

### �~�j�o�b�` : `makingData.py`

```python:makingData.py

def nextBatch(Otr,Ttr,Tlabel,batchSize,batchCnt = 0):
    ...
    
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
```
- ����
	- Otr: �w�K�f�[�^�̐����ϐ� $$x_1$$,$$x_2$$
	- Ttr: �w�K�f�[�^�̖ړI�ϐ� y
	- Tlabel: �w�K�f�[�^�̖ړI�ϐ��̃��x�� (one-hot)
	- batchSize: �o�b�`�T�C�Y
	- batchCnt: �o�b�`�J�E���g�̏����� (`trainingModel.py`�ōs��)
	


- �~�j�o�b�`�֐����Ă�
```python:trainingModel.py
# Get mini-batch
batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel,batchSize,batchCnt = 0)
```

<br>

<a id="ID_3-2"></a>


### Baseline Regression
- Optimizer�� `trainer_reg`�A�ړI�ϐ��̗\���l `reg_op`�A Loss�� `loss_reg`
- feed_dict�ŁA`x_reg`�ɐ����ϐ��A`y`�ɖړI�ϐ���^����

```python:makingData.py
        if methodModel == 0:
            _, trainPred, trainRegLoss = sess.run([trainer_reg, reg_op, loss_reg], feed_dict={x_reg:batchX, y:batchY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                testPred, testRegLoss = sess.run([reg_op_test, loss_reg_test], feed_dict={x_reg:xTest, y:yTest})
```        

<br>


<a id="ID_3-3"></a>

### Anchor-based

- Optimizer�� `trainer_cls`(����NN)�� `trainer_anc`(��ANN)�A�N���X�̒��S�l `pred_cls_center` �Ǝc���̗\���l `reg_op`�A Loss�� `loss_cls`(����NN)��`loss_anc`(��ANN)
- feed_dict�ŁA`x_cls`�ɐ����ϐ��A`x_reg`�ɃN���X�̒��S�l�Ɛ����ϐ���concat�������́A`y`�ɖړI�ϐ��A`y_label`�ɖړI�ϐ��̃��x��(one-hot)��^����
- �ړI�ϐ��̗\���l `trainPred` �� �N���X�̒��S�l `trainClsCenter` + �c���̗\���l `trainResPred` (python��̕\�L)


```python:makingData.py
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
```


<br>


<a id="ID_3-4"></a>
        
### ATR-Nets

- Optimizer�� `trainer_cls`(����NN)�� `trainer_atr`(��ANN)�A�N���X�̒��S�l `pred_cls_center` �Ɗg�債���c���̗\���l `reg_op`�A Loss�� `loss_cls`(����NN)��`loss_atr`(��ANN)��`loss_alpha`(alpha�w�K)
- feed_dict�ŁA`x_cls`�ɐ����ϐ��A`x_reg`�ɃN���X�̒��S�l�Ɛ����ϐ���concat�������́A`y`�ɖړI�ϐ��A`y_label`�ɖړI�ϐ��̃��x��(one-hot)��^����
- �ړI�ϐ��̗\���l `trainPred` �� �N���X�̒��S�l `trainClsCenter` + �g�傳�ꂽ�c�������Ƃɖ߂����\���l `trainRResPred` (python��̕\�L)



```python:makingData.py

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
```

<a id="ID_3-5"></a>

## ���f���̕ۑ�


```python:makingData.py
modelFileName = "model_{}_{}_{}_{}_{}_{}.ckpt".format(methodModel,sigma,nClass,pNum,nTrain,nTest)
modelPath = "models"
modelfullPath = os.path.join(modelPath,modelFileName)
saver.save(sess,modelfullPath)
```

<br>

<a id="ID_4"></a>


## ���s����: `plot.py`
�^�l��toydata�Ɨ\������toydata��visualization�f�B���N�g���ɁAloss��visualization\loss�f�B���N�g���ɕۑ������B


```python: plot.py
def Plot_3D(x1,x2,yGT,yPred,isPlot=False,methodModel=0,sigma=0,nClass=0,alpha=0,pNum=0,depth=0,isTrain=0):
    ...
    table = str.maketrans("", "" , string.punctuation + ".")
    sigma = str(sigma).translate(table)
    pdb.set_trace()
    if isPlot:
         
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        # �]���f�[�^plot
        ax.plot(np.reshape(x1,[-1,]),np.reshape(x2,[-1,]),np.reshape(yGT,[-1,]),"o",color="b",label="GT")
        # �\���lplot
        ax.plot(np.reshape(x1,[-1,]),np.reshape(x2,[-1,]),np.reshape(yPred,[-1,]),"o",color="r",label="Pred")
        plt.legend()
        fullPath = os.path.join(visualPath,"Pred_{}_{}_{}_{}_{}_{}_{}.png".format(methodModel,sigma,nClass,alpha,pNum,depth,isTrain))
        
        plt.savefig(fullPath)

#-----------------------------------------------------------------------------#              
def Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainRegLosses, testRegLosses, testPeriod, isPlot=False,methodModel=0,sigma=0,nClass=0,alpha=0,pNum=0,depth=0):
    if isPlot:
        if methodModel==2 or methodModel==1:
            # lossPlot
            plt.plot(np.arange(trainTotalLosses.shape[0]),trainTotalLosses,label="trainTotalLosses",color="r")
            plt.plot(np.arange(testTotalLosses.shape[0]),testTotalLosses,label="testTotalLosses",color="g")
            plt.plot(np.arange(trainClassLosses.shape[0]),trainClassLosses,label="trainClassLosses",color="b")
            plt.plot(np.arange(testClassLosses.shape[0]),testClassLosses,label="testClassLosses",color="k")
            plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,label="trainRegLosses",color="c")
            plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,label="testRegLosses",color="pink")
        
            plt.ylim([0,0.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            plt.legend()
            
            fullPath = os.path.join(visualPath,lossPath,"Loss_{}_{}_{}_{}_{}_{}.png".format(methodModel,sigma,nClass,alpha,pNum,depth))
        else:
            plt.plot(np.arange(trainClassLosses.shape[0]),trainClassLosses,label="trainRegLosses",color="c")
         
            plt.ylim([0,0.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            plt.legend()
            
            fullPath = os.path.join(visualPath,lossPath,"Loss_{}_{}_{}_{}_{}_{}.png".format(methodModel,sigma,nClass,alpha,pNum,depth))
        
        plt.savefig(fullPath)
#-----------------------------------------------------------------------------#      
```


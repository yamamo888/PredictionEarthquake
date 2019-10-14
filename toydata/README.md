# Tensorflow を用いた Baseline Regression とAnchor-based Regression と提案法 ATR-Nets の実装

この `README.md` には、各コードの実行結果、各コードの説明を記載しています。<br>
実行ファイルは `trainingModel.py`で、 toyデータ作成は `makingData.py`、nankaiデータの読み込みは`loadingData.py`、 結果の画像作成・出力は `plot.py` で、行っています。`makingData.py` と `loadingData.py`と`plot.py` は、実行ファイルから呼び出されます。<br>
以下、nankaiと表記する場合は、南海トラフシミュレーションデータを示す。<br>



## 項目 [Contents]

0. [使い方](#ID_0)
	1. [コマンド](#ID_0-1)

1. [使用するらせん階段データ : `makingData.py`](#ID_1)
	1. [コードの説明](#ID_1-1)
	2. [らせん階段の例(コードの実行結果)](#ID_1-2)

2. [nankaiデータの読み込み：`loadingData.py`](#ID_2)
	1. [学習・テストデータ](#ID_2-1)
	2. [評価データ：南海履歴データ](#ID_2-2)


3. [各手法のGraph作成 (tensorflow上) : `trainingMdel.py`](#ID_3)
	1. [パラメータ](#ID_3-1)
	2. [データの準備](#ID_3-2)
	3. [分類NNと回帰NN](#ID_3-3)
	4. [Anchor-based regressionとATR-Netsの回帰NNで使用する入力と出力の作成](#ID_3-4)
	5. [ATR-Netsの工夫点](#ID_3-5)
	6. [誤差関数・最適化](#ID_3-6)
	7. [関数の呼び出し](#ID_3-7)

4. [各手法のGraph実行 (python上) : `trainingMdel.py`](#ID_4)
	1. [ミニバッチ(学習データ) : `makingData.py`と`loadingData.py`](#ID_4-1)
	2. [Baseline Regression](#ID_4-2)
	3. [Anchor-based regression](#ID_4-3)
	4. [ATR-Nets](#ID_4-4)
	5. [モデルの保存](#ID_4-5)

5. [実行結果 : `Plot.py`](#ID_5)
	
<a id="ID_0"></a>

## 使い方

<a id="ID_0-1"></a>

### コマンド

以下はコマンド引数順と指定できる数字、コード上の名前と役割をまとめた。

|コマンド引数|指定可能数字|名前|役割|
|:----:|:----|:----|:----|
|1|0(ordinary regression) or 1(anchor-based) or 2(ATR-Nets)|methodModel|手法|
|2|float|sigma|toyデータのノイズ|
|3|10 or 20 or 50|nClass|分類のクラス数|
|4|2 or 3 or 5|pNum|toyデータの回転数|
|5|3 or 4 or 5|depth|回帰NNの層数|
|6|int|batchSize|バッチサイズ|
|7|int|nData|toyデータの数指定|
|8|float|trainRatio|toyデータの割合指定|
|9|float|alphaMode|alphaの初期値指定|
|10|0(toy mode) or 1(nanaki mode)|dataMode|toyかnankaiかの実験を選択|
|11|int|trialID|実験管理ID|

- 実験に必要がない引数も設定する必要あり。

- 例：nankaiでの実験設定 ---> モデルは Anchor-based Regression、クラス数は 10、3 階層回帰NNを使用、バッチサイズは1000、alphaの初期値は10の場合 :
```python trainingModel.py 1 0.00001 10 5 3 1000 1 1 0.1 1 1```

<br>

### コードの説明
- モデルの種類設定 `methodModel` は 0 のとき Ordinary Regression、1 のとき Anchor-based Regression、2 のとき ATR-Nets を実行する
- 目的変数のクラス数 `nClass` は 10,20,50がおすすめ
- 説明変数の回転数 `pNum`は 2か3か5ぐらいがおすすめ (1だと不定問題が起こらない、5以上は不定問題が起こりすぎるため)
- 回帰NNの層数 `depth`は 3,4,5


- コード上の変数とコマンド引数

```python:trainingModel.py
# -------------------------- command argment ----------------------------------
# Model type 0: ordinary regression, 1: anhor-based, 2: atr-nets
methodModel = int(sys.argv[1])
# noize of x1, x2
sigma = float(sys.argv[2])
# number of class
# nankai nClass = 11 or 21 or 51
nClass = int(sys.argv[3])
# number of rotation -> sin(pNum*pi) & cos(pNum*pi)
pNum = int(sys.argv[4])
# number of layer for Regression NN
depth = int(sys.argv[5])
# batch size
batchSize = int(sys.argv[6])
# data size
nData = int(sys.argv[7])
# rate of training
trainRatio = float(sys.argv[8])
# alpha
alphaMode = float(sys.argv[9])
# 0: toydata var., 1: nankai var.
dataMode = int(sys.argv[10])
# trial ID
trialID = sys.argv[11]
# -----------------------------------------------------------------------------

```

***

<a id="ID_1"></a>

## 使用するらせん階段データ : `makingData.py`
(x1,x2,y)からなる３次元のらせん階段データを作成する。
y は 0 ~ Sigma  の一様乱数分布 U(0,Sigma) に従って発生させたデータ。以下、目的変数yと説明変数 x1,x2 の関係式:<br>

![makedata](https://user-images.githubusercontent.com/32571202/63222016-9cd9dc00-c1dc-11e9-9379-637eb23210a5.png)

<br>

<a id="ID_1-1"></a>

### コードの説明

- らせん階段データ作成を行う。作成したデータを学習用データとテスト用データに分割する。分割割合は、`trainRatio`で指定し、学習用データ80%、テスト用データ20%

- 全データ数 `nData`、学習とテストの分割割合 `trainRatio` を決めると、学習データ数 `nTrain` とテストデータ数 `nTest` が決まる。例えば、`nData = 8000`のとき、`trainRatio = 0.8`とすると、学習データ数 6400, テストデータ数 1600となる。


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

- 引数の説明
	- x1,x2 の回転数 `pNum` と x1、x2の分散 `noise` は、 `trainingModel.py` を実行するときにコマンド引数で指定されたものが渡される。
	- 目的変数の範囲の最小値 `yMin` と最大値 `yMax` は、2,6。

- 次に、目的変数のラベル付けをする。分類ニューラルネットワークのある Anchor-based Regression と ATR-Netsの時に必要。


```python:makingData.py
def AnotationY(yMin=2,yMax=6,yClass=10,nClass=10,beta=1):
    ...
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
```
<br>

- データ作成時のパラメータの設定
	- クラス数 `nClass` は、 `trainingModel.py` を実行するときにコマンド引数で指定する。クラス分割数 `beta` は `trainingModel.py` で計算されたものが渡される。

<br>

<a id="ID_1-2"></a>

- らせん階段データ (全データ)
![toydata](https://user-images.githubusercontent.com/32571202/63222020-a2372680-c1dc-11e9-81d6-a85fa650ba21.png)

***

<a id="ID_2"></a>

## nankaiデータの読み込み：`loadingData.py`

<a id="ID_2-1"></a>

### 学習・テストデータ

- `namesInds`で読み取る学習pklデータが格納されたファイルを3つ指定、テストpklデータは常に同じファイルが読み込まれる
- `filePeriod`で読み取るファイルを変更するタイミングを指定

```python:loadingData.py
def loadTrainTestData(self,nameInds=[0,1,2,3,4]):
    # name of train pickles
    trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,8)]
    # name of test pickles
    testNames = ["b2b3b4b5b6_test1{}".format(self.nClass)]
```

```python:trainingModel.py
  if dataMode == 1:
        if i % filePeriod == 0:
            nameInds = random.sample(nametrInds,3) 
            myData.loadTrainTestData(nameInds=nameInds)
```

- 以下はテストデータの準備
- 

```python:loadingData.py
 #[number of data,]
self.xTest = self.xTest[:,1:6,:]
self.xTest = np.reshape(self.xTest,[-1,self.nCell*self.nWindow])
# test y
self.yTest = np.concatenate((self.y11Test[:,np.newaxis],self.y31Test[:,np.newaxis],self.y51Test[:,np.newaxis]),1)
# test label y
self.yTestLabel = np.concatenate((self.y11TestLabel[:,:,np.newaxis],self.y31TestLabel[:,:,np.newaxis],self.y51TestLabel[:,:,np.newaxis]),2)
```

***
<a id="ID_2-2"></a>

### 評価データ：南海履歴データ

```python:loadingData.py
def loadNankaiRireki(self):
        
    # nankai rireki path (slip velocity V)
    # nankaifeatue.pkl -> 190.pkl
    flag = False
    for fID in np.arange(256):

       nankairirekiPath = os.path.join(self.features,self.nankairireki,"{}_2.pkl".format(fID))
        
       with open(nankairirekiPath,"rb") as fp:
           x = pickle.load(fp)
	   if not flag:
               evalX = x
               flag = True
           else:
               evalX = np.vstack([evalX,x])

    self.evalX = evalX
```



***

<a id="ID_3"></a>

## 各手法による予測処理: `trainingModel.py`

Baseline Regression は、回帰ニューラルネットワーク (NN)であり、Anchor-based Regression は、分類NNと回帰NNを組み合わせたものであり、ATR-Netsは分類NNと回帰NNに、残差を拡大するネットワークを追加したものである。3つの手法を `trainingModel.py` にて1つにまとめている。3つ同時に実行することはできず、コマンド引数で`methodModel`を指定されたモデルが1つだけが実行される。実行方法はコマンドを参照。


- Anchor-basedのモデル図

![anchor-based](https://user-images.githubusercontent.com/32571202/63221987-571d1380-c1dc-11e9-84c2-2e1323c3a92e.png)
<br>

- ATR-Netsのモデル図

![atr-nets](https://user-images.githubusercontent.com/32571202/63222015-9a778200-c1dc-11e9-9aff-9a7460e21dd0.png)

***

<a id="ID_3-1"></a>

### パラメータ

- nankaiで使うパラメータ。<br>
	- nCell: 入力のセル数
	- nWindow: スライディングウィンドウのウィンドウ数
	- nkMax,nkMin: 南海領域の摩擦パラメータbの最小最大
	- tkMax,tkMin: 東南海、東海の摩擦パラメータbの最小最大
	- first_cls_center_nk: 分類NNで使う
	- nameInds: namesIndsをシャッフルして3つ選択、trainデータが入ってるデータを選択
	- filePeriod: nankaiのファイルをシャッフルするタイミングを決定

```python:trainingModel.py
# number of nankai cell(input)
nCell = 5
# number of sliding window
nWindow = 10
...
# maximum of nankai
nkMax = 0.0125
# minimum of nankai
nkMin = 0.017
# maximum of tonankai & tokai
tkMax = 0.012
# minimum of tonankai & tokai
tkMin = 0.0165
...
# Center variable of the first class in nankai
first_cls_center_nk = np.round(nkMin + (beta / 2),limitdecimal)
# Center variable of the first class in tonankai & tokai
first_cls_center_tk = np.round(tkMin + (beta / 2),limitdecimal)
...
# select nankai data(3/5) 
nametrInds = [0,1,2,3,4,5,6]
# random sample loading train data
nameInds = random.sample(nametrInds,3) 
...
# nankai file change timing
filePeriod = nTraining / 10
```

- toyとnankaiで共通のパラメータ
	- 分類NNと回帰NNの各種ノードの次元数
	- isSaveModel: モデルを保存するときTrue

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
...
# dropout
keepProbTrain = 1.0
# Learning rate
lr = 1e-3
# test count
testPeriod = 500
# if plot == True
isPlot = True
# if save model == True
isSaveModel = True

```

-toyとnankaiで異なるパラメータ
	- trainAlpha: alphaを学習するときTrue

```python:trainingModel.py
# Toy
if dataMode == 0:
    print(pNum)
    
    dInput = 2
    dOutput = 1
    # round decimal 
    limitdecimal = 3
    # Width class
    beta = np.round((yMax - yMin) / nClass,limitdecimal)
    dataName = f"toy_{trialID}"
    nTraining = 50000
    # not training alpha
    trainAlpha = True
# Nankai
else:
    dInput = nCell*nWindow
    dOutput = 3
    # round decimal 
    limitdecimal = 6
    # Width class
    beta = np.round((nkMax - nkMin) / nClass,limitdecimal)
    dataName = f"nankai_{trialID}"
    nTraining = 100000
    # not training alpha
    trainAlpha = False
```

***

<a id="ID_3-2"></a>

### データの準備


- `makingData.py`の`toyData`でtoyデータ作成・読み込み
- `loadingData.py`の`NankaiData`nankaiデータ読みこみ
	- `loadNankaiRireki()`は南海履歴データを読み込む関数

```python:trainingData.py
# --------------------------- data --------------------------------------------
# select toydata or nankaidata
if dataMode == 0:    
    myData = makingData.toyData(trainRatio=trainRatio, nData=nData, pNum=pNum, sigma=sigma)
    myData.createData(trialID,beta=beta)
else:
    myData = loadingNankai.NankaiData(nCell=nCell,nClass=nClass,nWindow=nWindow)
    myData.loadTrainTestData(nameInds=nameInds)
    #myData.loadNankaiRireki()
    if nClass == 10:
        nClass = 11
    elif nClass == 20:
        nClass = 21
    else:
        nClass =  51
``` 

***

<a id="ID_3-3"></a>

### 分類NNと回帰NN


- placeholder

```python:trainingModel.py
#------------------------- placeholder ----------------------------------------
# input of placeholder for classification
x_cls = tf.placeholder(tf.float32,shape=[None,dInput])
# input of placeholder for regression
x_reg = tf.placeholder(tf.float32,shape=[None,dInput])
x_reg_test = tf.placeholder(tf.float32,shape=[None,dInput])
# GT output of placeholder (target)
y = tf.placeholder(tf.float32,shape=[None,dOutput])
alpha_base = tf.placeholder(tf.float32)

if dataMode == 0:
    # GT output of label
    y_label = tf.placeholder(tf.int32,shape=[None,nClass])
else:
    y_label = tf.placeholder(tf.int32,shape=[None,nClass,dOutput])
```


- 重み(`weight_variable`)とバイアス(`bias_variable`)、sigmoid関数のalpha変数(`alpha_variable`)を定義する。`alpha` の初期値 `alphaInit` は平均 `mean` と分散 `stddev` を指定する必要あり。

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

- 引数
	- name: 変数の名前、例えば w_1,b_1,alpha
	- shape: 変数の形、例えば [128,128],[128],[1]

<br>

- 活性化関数と全結合を定義する。全結合 + sigmoid `fc_sigmoid`、全結合 + relu `fc_relu`、全結合のみ `fc`を定義。
- 入力は、入力 inputs、重み w、バイアス b、ドロップアウト数 keepProb を指定。
- keepProbはドロップアウトするノード数を指定できる。例えば、0.5 の場合は半分のノードしか使用されない。ただし、テスト時は1.0にしなければならない。


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

- 分類NN

	- {入力層: 2ノード、隠れ層1: 128ノード、隠れ層2: 128ノード、出力層: 1ノード} の分類NN (`Classify`) は、Anchor-based RegressionとATR-Netsで使用。
	- 入力 x は2次元行列で、出力 y はクラス数分の1次元ベクトル (0 ~ 1のクラス確率に対応)。
	- 各種ノード数はパラメータの項目を参照　



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
- 引数
	- テスト時は reuse=True にして、重みとバイアスを共有する。	

<br>

- 回帰NN
	- 回帰NN (`Regress`) はすべての手法で用いる。<font color="Red">ただし、ATR-Netsの時だけ出力層の活性化関数をsigmoidに指定(expを取るときにマイナス値は計算できないから)。

	- {入力層: 2ノード or 3ノード、隠れ層1: 128ノード、隠れ層2: 128ノード、出力層: 1ノード} の回帰NN (`Regress`)

	- 入力 `x_reg` は、Baseline Regressionの時は2次元行列の説明変数で、Anchor-based regressionとATR-Netsの時は2次元行列の説明変数と1次元のクラスの中心値とをconcatした3次元行列である。

	- 出力は、Baseline Regression の時は1次元ベクトルの目的変数の予測値で、Anchor-based regressionとATR-Netsの時は1次元ベクトルの残差(=真値とクラスの中心値)である。

	- 各種ノード数はパラメータの項目を参照　


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

- 回帰NNと同様の構造で、入力がresidualの残差回帰NN関数(割愛)。

```python:trainingModel.py
def ResidualRegress(x_reg,reuse=False,isATR=False,depth=0,keepProb=1.0):
...
```

- 引数
	- isATR: ATR-Netsの時は True、その他の手法は False
	- depth: 階層を指定　例えば depth=3 のときは 3 階層モデル 現在は 3,4,5 階層モデルに対応
	- テスト時は reuse=True にして、重みとバイアスを共有する。

***

<a id="ID_3-4"></a>

### Anchor-based regressionとATR-Netsの回帰NNで使用する入力と出力の作成

- 回帰NNの入力
	- クラス確率のベクトルの最大確率のクラス `pred_maxcls` から、そのクラスの中心値 `pred_cls_center` を取得。説明変数とconcatする。
- 回帰NNの出力の真値
	- 残差 (目的変数 - クラスの中心値 `pred_cls_center` )

- nankaiのときは領域によって、刻み開始数値が異なる。南海は`first_cls_center_nk`、東南海・東海は`first_cls_center_tk`
- 南海履歴データを使う時は、真値yがなくresidualの計算ができないので、分岐させている

```python:trainingModel.py
def CreateRegInputOutput(x,y,cls_score,isEval=False):
    ...
    
     if dataMode == 0:

        # Max class of predicted class
        pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
        # Center variable of class        
        pred_cls_center = pred_maxcls * beta + first_cls_center
    
    else:
        # Max class of predicted class
        pred_maxcls1 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,0],axis=1),tf.float32),1)  
        # Max class of predicted class
        pred_maxcls2 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,1],axis=1),tf.float32),1)  
        # Max class of predicted class
        pred_maxcls3 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,2],axis=1),tf.float32),1)

        # Center variable of class for nankai       
        pred_cls_center1 = pred_maxcls1 * beta + first_cls_center_nk
        # Center variable of class for tonaki        
        pred_cls_center2 = pred_maxcls2 * beta + first_cls_center_tk
        # Center variable of class for tokai       
        pred_cls_center3 = pred_maxcls3 * beta + first_cls_center_tk
        # [number of data, cell(=3)] 
        pred_cls_center = tf.concat((pred_cls_center1,pred_cls_center2,pred_cls_center3),1)
    
    
    if isEval:
        return pred_cls_center, cls_center_x
    else:
        
        # residual = objective - center variavle of class 
        r = y - pred_cls_center
        # feature vector + center variable of class
        cls_center_x =  tf.concat((pred_cls_center,x),axis=1)
        
        return pred_cls_center, r, cls_center_x
```

- 引数
	- x: 説明変数 x = x1 + x2
	- y: 目的変数
	- cls_score: クラス確率 (`Classify` の出力)
	- isEval: 南海履歴データを用いるとき True


***
<a id="ID_3-5"></a>

### ATR-Netsの工夫点

- 残差の範囲を **sigmoid関数** を用いて、[0,1] にエンコードする
- sigmoid関数の傾き `alpha` は学習して最適化する
- 残差とエンコードされた残差との関係式： (未定) <bf>

```python:trainingModel.py
def TruncatedResidual(r,alpha_base,reuse=False):
    ...
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        # traincated adjustment parameter
        alpha = alpha_variable("alpha",[1]) 
        # trauncated range of residual
        r_at = 1/(1 + tf.exp(- alpha * r))
        
        return r_at, alphawith tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        
        alpha = alpha_variable("alpha",[dOutput]) 
        alpha_final = tf.multiply(alpha,alpha_base)
        
        if trainAlpha:
            r_at = 1/(1 + tf.exp(- alpha_final * r))
            return r_at, alpha_final
        else:
            r_at = 1/(1 + tf.exp(- alpha * r))
            return r_at, alpha
#-----------------------------------------------------------------------------#
```

- 引数
	- r: 真の残差
	- alpha_base:

- エンコードされた残差をもとの範囲の残差に戻す 
- 式： (未定) <bf>

```python:trainingModel.py
def Reduce(r_at,param,reuse=False):
    ...
   with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()

        #pred_r = (-1/param) * tf.log((1/r_at) - 1)
        pred_r = 1/param * tf.log(r_at/(1-r_at + 1e-8))
        
        return pred_r
#-----------------------------------------------------------------------------# 
```

- 引数
	- r_at: エンコードされた残差
	- param: sigmoid関数の傾き


***
<a id="ID_3-6"></a>

### 損失関数・最適化

- 分類NNはクロスエントロピー誤差で、回帰NNとalphaを学習するときは絶対平均誤差
- **1つの学習器を学習するときは他の学習器を frozenする。そのため、name_scopeで学習する学習器を指定する必要あり。例えば、分類NNを学習するときは、他の回帰NN学習器とalpha学習器を学習しない (`name_scope="Regress"` を指定)**

```python:trainingModel.py
def Loss(y,predict,isCE=False):
    ...
   if isCE:
        if dataMode == 0:
            return tf.losses.softmax_cross_entropy(y,predict)
        else:
            return tf.losses.softmax_cross_entropy(y[:,:,0],predict[:,:,0]) + tf.losses.softmax_cross_entropy(y[:,:,1],predict[:,:,1]) + tf.losses.softmax_cross_entropy(y[:,:,2],predict[:,:,2])
    else:
        return tf.reduce_mean(tf.square(y - predict))
```

```python:trainingModel.py
def Optimizer(loss,name_scope="Regress"):
    ...
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
    opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=Vars)
    return opt

```

- 引数
	- isCE: 回帰NNとalpha学習する時は False、分類NNの時は True (default)
	- name_scope: 回帰NNの時は Regress (default)、分類NNの時は Classify、alpha学習するときは TrResidual を指定　

***
<a id="ID_3-7"></a>

### 関数の呼び出し

- 分類NN
	- 南海履歴データを使う時は`cls_op_eval`のコメントアウトを外す

```python:trainingModel.py
    cls_op = Classify(x_cls)
    cls_op_test = Classify(x_cls,reuse=True)
    #cls_op_eval = Classify(x_cls,reuse=True,keepProb=1.0)
```

- Anchor-based regressionとATR-Netsの回帰NNでは、説明変数 x とクラスの中心値 `pred_cls_center` を concatした`reg_in` を入力とし、残差 (目的変数 - クラスの中心値 ) `res` を出力の真値にする。


```python:trainingModel.py
    pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,cls_op)
    pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,cls_op_test)
```

- 回帰NNと残差回帰NN
	- Baseline regression の入力は説明変数 `x_reg`、Anchor-based regressionとATR-Netsの入力は説明変数 x とクラスの中心値 `reg_in`する 
	- Baseline regressionとAnchor-based regressionの時 `isATR`は False、ATR-Netsの時 True (出力層の活性化関数をsigmoidにするため)

```python:trainingModel.py
reg_res = ResidualRegress(reg_in,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_res_test = ResidualRegress(reg_in_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)
#reg_res_eval = ResidualRegress(reg_in_eval,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)

reg_y = Regress(x_reg,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_y_test = Regress(x_reg_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)
#reg_y_eval = Regress(x_reg_eval,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)
```    

- L1とL2損失(今はコメントアウト)
	- どのdepth数でも共通でパスが通るように、`w1_reg` と `w2_reg` しか取り出してない

```python:trainingModel.py
#res_l1 = tf.reduce_sum(tf.abs(ResidualRegress.w1_reg)) + tf.reduce_sum(tf.abs(ResidualRegress.w2_reg))
#reg_l1 = tf.reduce_sum(tf.abs(Regress.w1_reg)) + tf.reduce_sum(tf.abs(Regress.w2_reg))

# l2
#res_l2 = tf.nn.l2_loss(ResidualRegress.w1_reg) + tf.nn.l2_loss(ResidualRegress.w2_reg)
#reg_l2 = tf.nn.l2_loss(Regress.w1_reg) + tf.nn.l2_loss(Regress.w2_reg)
```

- Adaptive Truncated residual 
- 入力は真の残差 `res`、出力は真の拡大された残差 `res_atr` とsigmoid関数の傾き `alpha_op`

```python:trainingModel.py
res_atr, alpha = TruncatedResidual(res,alpha_base)
res_atr_test, alpha_test = TruncatedResidual(res_test,alpha_base,reuse=True)
```

- Reduce residual
- 入力は予測した拡大残差 `res_op`とsigmoid関数の傾き `alpha_op`、出力は元に戻した残差 `reduce_res`
- TruncatedResidual()内で定義されたalphaを使用するので、学習・テスト両方とも `reuse=True`

```python:trainingModel.py
reduce_res_op = Reduce(reg_res,alpha,reuse=True)
reduce_res_op_test = Reduce(reg_res_test,alpha_test,reuse=True)
```

- 予測した目的変数 `pred_y` = クラスの中心値 `pred_cls_center` + 元に戻した残差 `reduce_res_op`

```python:trainingModel.py
# predicted y by ATR-Nets
pred_y = pred_cls_center + reduce_res_op
pred_y_test = pred_cls_center_test + reduce_res_op_test
```

- 誤差関数
	- Baseline Regressionは `loss_reg`、Anchor-based regressionは `loss_cls`と`loss_anc`、ATR-Netsは`loss_cls`と`loss_atr`と`loss_alpha`
	- `loss_cls`は `isCE=True`
	- `loss_cls`はクラスのラベル、 `loss_reg`は目的変数、`loss_anc`は残差、`loss_atr`はエンコードされた残差が真値である。`loss_cls`は分類NNの出力 `cls_op`で`loss_reg`、`loss_anc`、`loss_atr`は回帰NNの出力 `reg_op` を使用する。
	- l1とl2損失を実装したいときは上記の`reg_l1`、`res_l1`、`reg_l2`、`res_l2`を誤差関数に付け足す。


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

- 最適化
	- Baseline Regressionは `trainer_reg`、Anchor-based regressionは`trainer_cls`と`trainer_anc`、ATR-Netsは`trainer_cls`と`trainer_atr`と`trainer_alpha` 
	- name_scopeで分類NNは`Classify`、alpha学習器は`TrResidual`を指定。(回帰NNは指定する必要なし)

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

***
<a id="ID_4"></a>

## 各手法のGraph実行 (python上) : `trainingMdel.py`
ミニバッチデータ取得、学習フェーズ実行、テストフェーズ実行の3つの段階に大きく分けられる。

<a id="ID_4-1"></a>

### ミニバッチ : `makingData.py`と`loadingData.py`

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
- 引数
	- Otr: 学習データの説明変数 $$x_1$$,$$x_2$$
	- Ttr: 学習データの目的変数 y
	- Tlabel: 学習データの目的変数のラベル (one-hot)
	- batchSize: バッチサイズ
	- batchCnt: バッチカウントの初期化 (`trainingModel.py`で行う)
	
- ミニバッチ関数を呼ぶ
```python:trainingModel.py
# Get mini-batch
batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel,batchSize,batchCnt = 0)
```

- 以下、nankaiのデータ読み込み
	- 真値yも5つ(`*1*Train`,`*2*Train`,`*3*Train`,`*4*Train`,`*5*Train`)あるうちから、3つ(`*1*Train`,`*3*Train`,`*5*Train`)を選択

```python:loadingData.py
def nextBatch(self,batchSize):
        
        sInd = batchSize * self.batchCnt
        eInd = sInd + batchSize
       
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        trX = np.concatenate((self.x11Train[:,1:6,:],self.x12Train[:,1:6,:],self.x13Train[:,1:6,:]),0) 
        # mini-batch, [number of data, cell(=5)*dimention of features(=10)]
        batchX = np.reshape(trX[sInd:eInd],[-1,self.nCell*self.nWindow])
        # test all targets
        trY1 = np.concatenate((self.y11Train,self.y12Train,self.y13Train),0)
        trY2 = np.concatenate((self.y31Train,self.y32Train,self.y33Train),0)
        trY3 = np.concatenate((self.y51Train,self.y52Train,self.y53Train),0)
        # [number of data(mini-batch), cell(=3)] 
        batchY = np.concatenate((trY1[sInd:eInd,np.newaxis],trY2[sInd:eInd,np.newaxis],trY3[sInd:eInd,np.newaxis]),1)
        
        # train all labels, trlabel1 = nankai
        trlabel1 = np.concatenate((self.y11TrainLabel,self.y12TrainLabel,self.y13TrainLabel),0)
        trlabel2 = np.concatenate((self.y31TrainLabel,self.y32TrainLabel,self.y33TrainLabel),0)
        trlabel3 = np.concatenate((self.y51TrainLabel,self.y52TrainLabel,self.y53TrainLabel),0)
        # [number of data, number of class(self.nClass), cell(=3)] 
        batchlabelY = np.concatenate((trlabel1[sInd:eInd,:,np.newaxis],trlabel2[sInd:eInd,:,np.newaxis],trlabel3[sInd:eInd,:,np.newaxis]),2)
        
        if eInd + batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1

        return batchX, batchY, batchlabelY
```

***
<a id="ID_4-2"></a>


### Baseline Regression
- Optimizerは `trainer_reg`、目的変数の予測値 `reg_op`、 Lossは `loss_reg`
- feed_dictで、`x_reg`に説明変数、`y`に目的変数を与える

```python:makingData.py
        if methodModel == 0:
            _, trainPred, trainRegLoss = sess.run([trainer_reg, reg_op, loss_reg], feed_dict={x_reg:batchX, y:batchY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                testPred, testRegLoss = sess.run([reg_op_test, loss_reg_test], feed_dict={x_reg:xTest, y:yTest})
```       

***
<a id="ID_4-3"></a>

### Anchor-based

- Optimizerは `trainer_cls`(分類NN)と `trainer_anc`(回帰NN)、クラスの中心値 `pred_cls_center` と残差の予測値 `reg_op`、 Lossは `loss_cls`(分類NN)と`loss_anc`(回帰NN)
- feed_dictで、`x_cls`に説明変数、`x_reg`にクラスの中心値と説明変数をconcatしたもの、`y`に目的変数、`y_label`に目的変数のラベル(one-hot)を与える
- 目的変数の予測値 `trainPred` は クラスの中心値 `trainClsCenter` + 残差の予測値 `trainResPred` (python上の表記)


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
***
<a id="ID_4-4"></a>
        
### ATR-Nets

- Optimizerは `trainer_cls`(分類NN)と `trainer_atr`(回帰NN)、クラスの中心値 `pred_cls_center` と拡大した残差の予測値 `reg_op`、 Lossは `loss_cls`(分類NN)と`loss_atr`(回帰NN)と`loss_alpha`(alpha学習)
- feed_dictで、`x_cls`に説明変数、`x_reg`にクラスの中心値と説明変数をconcatしたもの、`y`に目的変数、`y_label`に目的変数のラベル(one-hot)を与える
- 目的変数の予測値 `trainPred` は クラスの中心値 `trainClsCenter` + 拡大された残差をもとに戻した予測値 `trainRResPred` (python上の表記)



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

***
<a id="ID_4-5"></a>

## モデルの保存

すべてのモデルで、以下のようにモデル保存

```python:makingData.py
if isSaveModel:
    saver.save(sess,os.path.join(modelPath,savePath,"model_{}_{}_{}_{}.ckpt".format(methodModel,dataName,nClass)),global_step=i)
```

また、最終的な値はpklデータとして`results\*pickles`に保存

```python:trainingData.py
with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
            pickle.dump(batchY,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(myData.yTest,fp)
            pickle.dump(testPred,fp)
            pickle.dump(trainRegLosses,fp)
            pickle.dump(trainTotalVar,fp)
            pickle.dump(testRegLosses,fp)
            pickle.dump(testTotalVar,fp)
```


***
<a id="ID_5"></a>

## 実行結果: `Plot.py`
- テストデータの真値のtoyと予測したtoyは、`visualization` に保存
- lossは、`visualization\loss` に保存
- テストデータのnankaiの分散は、`visualization\scatter`
- 保存するファイル名を変えたいときは`savefilePath`を変えてください
	- ex) savefilePath = "{}_{}_{}_{}_{}_{}_{}".format(dataName,methodModel,nClass,depth,batchSize,testAlpha,trialID)

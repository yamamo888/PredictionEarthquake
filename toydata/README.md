# Tensorflow を用いた Baseline Regression とAnchor-based Regression と提案法 ATR-Nets の実装

この `README.md` には、各コードの実行結果、各コードの説明を記載しています。
実行ファイルは `trainingModel.py`で、 データ作成は `makingData.py`、 結果の画像作成・出力は `plot.py` で行っています。`makingData.py` と `plot.py` は、実行ファイルから呼び出されます。


## 項目 [Contents]

0. [使い方](#ID_0)
	1. [コマンド](#ID_0-1)

1. [使用するらせん階段データ: `makingData.py`](#ID_1)
	1. [コードの説明](#ID_1-1)
	2. [らせん階段の例(コードの実行結果)](#ID_1-2)

2. [各手法のGraph作成 (tensorflow上): `trainingMdel.py`](#ID_2)
	1. [パラメータ](#ID_2-1-1)
	2. [分類NNと回帰NN](#ID_2-1-2)
	3. [Anchor-based regressionとATR-Netsの回帰NNで使用する入力と出力の作成](#ID_2-1-3)
	4. [ATR-Netsの工夫点](#ID_2-1-4)
	5. [関数の呼び出し](#ID_2-1-5)
	6. [誤差関数・最適化](#ID_2-1-6)

3. [各手法のGraph実行 (python上): `trainingMdel.py`](#ID_3)
	0. [ミニバッチ(学習データ): `makingData.py`](#ID_3-1)
	1. [Baseline Regression](#ID_3-2)
	2. [Anchor-based regression](#ID_3-3)
	3. [ATR-Nets](#ID_3-4)
	4. [モデルの保存](#ID_3-5)
3. [実行結果](#ID_3)

<a id="ID_0"></a>

## 使い方

<a id="ID_0-1"></a>

### コマンド
```
python trainingModel.py <モデルの種類(methodModel)> <ノイズ(sigma)> <クラス数(number of class)> <回転数(number of rotation)> <階層数(number of layer in Regression NN)>
```
モデルは Anchor-based Regression、説明変数の分散は 0.00001、クラス数は 10、回転数 5、3階層回帰NNを使用したい場合:<br>
```python trainingModel.py 1 0.00001 10 5 3```

<br>

### コードの説明
- モデルの種類設定 `methodModel` は 0 のとき Baseline Regression、1 のとき Anchor-based Regression、2 のとき ATR-Nets を実行する
- 説明変数の分散 `sigma` は 0.0000001 以下がおすすめ ($x_1$,$x_2$ の大きさが小さいので)
- 目的変数のクラス数 `nClass` は 10,20,50がおすすめ
- 説明変数の回転数 `pNum`は 2か3か5ぐらいがおすすめ (1だと不定問題が起こらない、5以上は不定問題が起こりすぎるため)
- 回帰NNの層数 `depth`は 3,4,5

<br>

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

## 使用するらせん階段データ : `makingData.py`
($x_1$,$x_2$,y)からなる３次元のらせん階段データを作成する。
$y^n$は、0 ~ $Sigma$ の一様乱数分布 U(0, $\Sigma$) に従って発生させたデータ。以下、目的変数yと説明変数$x_1$,$x_2$の関係式:<br>
```math
y^n &\overset{\mathrm{i.i.d}{\sim}}U(y_\mathrm{min},y_\mathrm{max})
x_1^n &= sin(m \times y^n) + \frac{1}{\log(y^n)} + \mathcal{N}(0,\Sigma)
x_2^n &= cos(m \times y^n) + \frac{1}{\log(y^n)} + \mathcal{N}(0,\Sigma)
```
<br>

<a id="ID_1-1"></a>

### コードの説明

- まず、らせん階段データ作成を行う。作成したデータを学習用データとテスト用データに分割する。分割割合は、`trainRatio`で指定し、学習用データ80%、テスト用データ20%

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
	- $x_1$,$x_2$の回転数 `pNum` と x1、x2の分散 `noise` は、 `trainingModel.py` を実行するときにコマンド引数で指定されたものが渡される。
	- 目的変数の範囲の最小値 y_\mathrm{min}と最大値 y_\mathrm{max}は、2,6。


<br>

- 次に、目的変数のラベル付けをする。<font color="Red">分類ニューラルネットワークのある Anchor-based Regression と ATR-Netsの時に必要。</font>

```python:makingData.py
def AnotationY(yMin=2,yMax=6,yClass=10,nClass=10,beta=1):
    ...

    # Get only target variables Y ()
    _,_,_,_,_,_, target = SplitTrainTest()
    
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
![toydata](/results/toydata.png)

<br>

<a id="ID_1-1"></a>

## 各手法による予測処理: `trainingMdel.py`
Baseline Regression は、回帰ニューラルネットワーク (NN)であり、Anchor-based Regression は、分類NNと回帰NNを組み合わせたものであり、ATR-Netsは分類NNと回帰NNに、残差を拡大するネットワークを追加したものである。3つの手法を `trainingModel.py` にて1つにまとめている。3つ同時に実行することはできず、コマンド引数で`methodModel`を指定されたモデルが1つだけが実行される。実行方法はコマンドを参照。 <a id="ID_0-1"></a>

<br>

<a id="ID_2-1-1"></a>

### パラメータ

```python:trainingModel.py
# --------------------------- parameters --------------------------------------
# Number of node dimention in Classification
nInput = xTrain.shape[0]
dInput = xTest.shape[1]
nHidden = 128 # node of 1 hidden
nHidden2 = 128 # node of 2 hidden 
# round decimal 
limitdecimal = 3
# Width class
beta = np.round((yMax - yMin) / nClass,limitdecimal)
# Center variable of the first class
first_cls_center = np.round(yMin + (beta / 2),limitdecimal)

# Number of node dimention in Regression ##
nRegOutput = 1
if methodModel == 0:
    nRegInput = dInput
else:
    nRegInput = nRegOutput + dInput
nRegHidden = 128 # node of 1 hidden
nRegHidden2 = 128 # node of 2 hidden
nRegHidden3 = 128 # node of 3 hidden
nRegHidden4 = 128 # node of 4 hidden
nRegOutput = 1

# Learning rate
lr = 1e-4
# number of training
nTraining = 300000
# test count
testPeriod = 500
# if plot == True
isPlot = True
# -----------------------------------------------------------------------------
```

<br>

- `makingData.py`から説明変数と目的変数(学習とテストの両方)とを受け取り、説明変数の x は $x_1$,$x_2$ を concat して2次元のデータにする。

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
- keepProbはドロップアウトするノード数を指定できる。例えば、0.5の場合は半分のノードしか使用されない。ただし、テスト時は1.0にしなければならない。


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

### 分類NNと回帰NN

- {入力層: 2ノード、隠れ層1: 128ノード、隠れ層2: 128ノード、出力層: 1ノード} の分類NN (`Classify`) は、Anchor-based RegressionとATR-Netsで使用。
-入力 x は2次元行列で、出力 y はクラス数分の1次元ベクトル (0 ~ 1のクラス確率に対応)。
-各種ノード数はパラメータの項目を参照　<a id="ID_0-1"></a>



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

- 回帰NN (`Regress`) はすべての手法で用いる。<font color="Red">ただし、ATR-Netsの時だけ出力層の活性化関数をsigmoidに指定(expを取るときにマイナス値は計算できないから)。
-入力 x は、Baseline Regressionの時は2次元行列の説明変数で、Anchor-based regressionとATR-Netsの時は2次元行列の説明変数と1次元のクラスの中心値とをconcatした3次元行列である。出力は y は、Baseline Regression の時は1次元ベクトルの目的変数の予測値で、Anchor-based regressionとATR-Netsの時は1次元ベクトルの残差(=真値とクラスの中心値)である。
-各種ノード数はパラメータの項目を参照　<a id="ID_0-1"></a>


```python:trainingModel.py
def Regress(x_reg,reuse=False,isR=False,depth=0):
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
            
            if isR:
                # shape=[None,number of dimention (y)]
                return fc(h1,w2_reg,bias2_reg,keepProb)
            else:
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb)
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegOutput])
            bias3_reg = bias_variable('bias3_reg',[nRegOutput])
            
            if isR:
                return fc(h2,w3_reg,bias3_reg,keepProb)
            else:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb)
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
            
            if isR:
                return fc(h3,w4_reg,bias4_reg,keepProb) 
            else:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb) 
#-----------------------------------------------------------------------------#
```
<br>

- 引数
	- isR: ATR-Netsの時は True、その他の手法は False
	- depth: 階層を指定　例えば depth=3 のときは 3 階層モデル 現在は 3,4,5 階層モデルに対応
	- テスト時は reuse=True にして、重みとバイアスを共有する。

<br>



<a id="ID_2-1-3"></a>


### Anchor-based regressionとATR-Netsの回帰NNで使用する入力と出力の作成

- 回帰NNの入力
	- クラス確率のベクトルの最大確率のクラス `pred_maxcls` から、そのクラスの中心値 `pred_cls_center` を取得。説明変数とconcatする。
- 回帰NNの出力の真値
	- 残差 (目的変数 - クラスの中心値 `pred_cls_center` )

```python:trainingModel.py
def CreateRegInputOutput(x,cls_score):
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

- 引数
	- x: 説明変数
	- cls_score: クラス確率 (`Classify` の出力)


<br>

<a id="ID_2-1-4"></a>

### ATR-Netsの工夫点

- 残差の範囲を_sigmoid関数_を用いて、[0,1] にエンコードする
- sigmoid関数の傾き `alpha` は学習して最適化する
- 残差とエンコードされた残差との関係式： $\textbf{r_at}= \frac{1}{1 + exp{-\alpha \textbf{r}}}$


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



- 引数
	- r: 真の残差

<br>

- 残差とエンコードされた残差の関係図 
![atr](/results/atr.png)

<br>

- エンコードされた残差をもとの範囲の残差に戻す 
- 式： $\textbf{r} = \frac{-1}{\alpha}\log{\frac{1}{\textbf{r_at}} - 1 }$ (残差とエンコードされた残差の関係式と逆)


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

- 引数
	- r_at: エンコードされた残差
	- alpha: sigmoid関数の傾き

<br>

<a id="ID_2-1-4"></a>

### 損失関数・最適化

- 分類NNはクロスエントロピー誤差で、回帰NNとalphaを学習するときは絶対平均誤差
- 1つの学習器を学習するときは他の学習器を frozenする。そのため、name_scopeで学習する学習器を指定する必要あり。例えば、分類NNを学習するときは、他の回帰NN学習器とalpha学習器を学習しない (`name_scope="Regress"` を指定)

```python:trainingModel.py
def Loss(y,predict,isR=False):
    ...
    if isR:
        return tf.reduce_mean(tf.abs(y - predict))
    else:
        return tf.losses.softmax_cross_entropy(y,predict)
#-----------------------------------------------------------------------------#
def Optimizer(loss,name_scope="Regress"):
    ...
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
    opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=Vars)
    return opt

```

- 引数
	- isR: 回帰NNとalpha学習する時は True、分類NNの時は False
	- name_scope: 回帰NNの時は Regress、分類NNの時は Classify、alpha学習するときは TrResidual を指定　

<br>


<a id="ID_2-1-5"></a>

### 関数の呼び出し

- 分類NN

```python:trainingModel.py
    cls_op = Classify(x_cls)
    cls_op_test = Classify(x_cls,reuse=True)
```

- Anchor-based regressionとATR-Netsの回帰NNでは、説明変数 x とクラスの中心値 `pred_cls_center` を concatした`reg_in` を入力とし、残差 (目的変数 - クラスの中心値 ) `res` を出力の真値にする。


```python:trainingModel.py
    pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,cls_op)
    pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,cls_op_test)
```

- 回帰NN
	- Baseline regression の入力は説明変数 `x_reg`、Anchor-based regressionとATR-Netsの入力は説明変数 x とクラスの中心値 `reg_in`
	- Baseline regressionとAnchor-based regressionの時 `is_atr`は False、ATR-Netsの時 True (出力層の活性化関数をsigmoidにするため)

```python:trainingModel.py
# Baseline regression
reg_op = Regress(x_reg,isATR=is_atr,depth=depth)
reg_op_test = Regress(x_reg,reuse=True,isATR=is_atr,depth=depth)
    
# Anchor-based regression & ATR-Nets
res_op = Regress(reg_in,isATR=is_atr,depth=depth)
res_op_test = Regress(reg_in_test,reuse=True,isATR=is_atr,depth=depth)
```    

- Adaptive Truncated residual 
- 入力は真の残差 `res`、出力は真の拡大された残差 `res_atr` とsigmoid関数の傾き `alpha_op`


```python:trainingModel.py
res_atr, alpha_op = TruncatedResidual(res)
res_atr_test, alpha_op_test = TruncatedResidual(res_test,reuse=True)
```
- Reduce residual
- 入力は予測した拡大残差 `res_op`とsigmoid関数の傾き `alpha_op`、出力は元に戻した残差 `reduce_res`  

```python:trainingModel.py
reduce_res = Reduce(res_op,alpha_op)
reduce_res_test = Reduce(res_op_test,alpha_op_test,reuse=True)
```
<br>

- 予測した目的変数 `pred_y` = クラスの中心値 `pred_cls_center` + 元に戻した残差 `reduce_res_op`


```python:trainingModel.py
# predicted y by ATR-Nets
pred_y = pred_cls_center + reduce_res_op
pred_y_test = pred_cls_center_test + reduce_res_op_test
```
<br>

- 誤差関数
- Baseline Regressionは `loss_reg`、Anchor-based regressionは `loss_cls`と`loss_anc`、ATR-Netsは`loss_cls`と`loss_atr`と`loss_alpha`
- `loss_cls`以外は`is_reg`をTrue指定
- `loss_cls`の時　クラスのラベル

```python:trainingModel.py
# Classification loss
# gt label (y_label) vs predicted label (cls_op)
loss_cls = Loss(y_label,cls_op)
loss_cls_test = Loss(y_label,cls_op_test)
    
# Baseline regression loss train & test
# gt value (y) vs predicted value (reg_op)
loss_reg = Loss(y,reg_op,isR=is_reg)
loss_reg_test = Loss(y,reg_op_test,isR=is_reg)
    
# Regression loss for Anchor-based
# gt residual (res) vs predicted residual (res_op)
loss_anc = Loss(res,res_op,isR=is_reg)
loss_anc_test = Loss(res_test,res_op_test,isR=is_reg)
    
# Regression loss for Atr-nets
# gt truncated residual (res_at) vs predicted truncated residual (res_op)
loss_atr = Loss(res_atr,res_op,isR=is_reg)
loss_atr_test = Loss(res_atr_test,res_op_test,isR=is_reg)
    
# Training alpha loss
# gt value (y) vs predicted value (pred_yz = pred_cls_center + reduce_res)
loss_alpha = Loss(y,pred_y,isR=is_reg)
loss_alpha_test = Loss(y,pred_y_test,isR=is_reg)
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
<br>

<a id="ID_3"></a>


## 各手法のGraph実行 (python上): `trainingMdel.py`
ミニバッチデータ取得、学習フェーズ実行、テストフェーズ実行の3つの段階に大きく分けられる。

<br>



<a id="ID_3-1"></a>

### ミニバッチ

```python:makingData.py

def nextBatch(Otr,Ttr,Tlabel,batchSize):
    ...
    # batch count initialize
    #batchCnt = 0
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
	- Otr: 学習データの説明変数 $x_1$,$x_2$
	- Ttr: 学習データの目的変数 y
	- Tlabel: 学習データの目的変数のラベル (one-hot)
	- batchSize: バッチサイズ
	


- ミニバッチ関数を呼ぶ
```python:trainingModel.py
# Get mini-batch
batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel)

<br>

<a id="ID_3-2"></a>


### Baseline Regression
- Optimizer `trainer_reg`、目的変数の予測値 `res_op`、 Loss `loss_reg`
- feed_dictで、`x_reg`に説明変数、`y`に目的変数を与える

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
- Optimizer `trainer_cls`(分類NN)と `trainer_anc`(回帰NN)、クラスの中心値 `pred_cls_center` と残差の予測値 `res_op`、 Loss `loss_cls`(分類NN)と`loss_anc`(回帰NN)
- feed_dictで、`x_cls`に説明変数、`y`に目的変数、`y_label`に目的変数のラベル(one-hot)を与える
- 目的変数の予測値 `trainPred` は クラスの中心値 `trainClsCenter` + 残差の予測値 `trainResPred` (python上の表記)


```python:makingData.py
elif methodModel == 1:
            # classication
            _, trainClsCenter, trainClsLoss = sess.run([trainer_cls, pred_cls_center, loss_cls], feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            # regression
            _, trainResPred, trainResLoss = sess.run([trainer_anc, res_op, loss_anc],feed_dict={y:batchY, y_label:batchlabelY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                # classication
                testClsLoss, testClsCenter = sess.run([loss_cls_test, pred_cls_center_test], feed_dict={x_cls:xTest, y:yTest, y_label:yTestlabel})    
                # regression
                testResLoss, testResPred = sess.run([loss_anc_test, res_op_test], feed_dict={y:yTest, y_label:yTestlabel})
                
                # Reduce
                trainPred = trainClsCenter + trainResPred
                testPred = testClsCenter + testResPred     
```


<br>


<a id="ID_3-4"></a>
        
### ATR-Nets

- Optimizer `trainer_cls`(分類NN)と `trainer_atr`(回帰NN)、クラスの中心値 `pred_cls_center` と拡大した残差の予測値 `res_op`、 Loss `loss_cls`(分類NN)と`loss_atr`(回帰NN)と`loss_alpha`(alpha学習)
- feed_dictで、`x_cls`に説明変数、`y`に目的変数、`y_label`に目的変数のラベル(one-hot)を与える
- 目的変数の予測値 `trainPred` は クラスの中心値 `trainClsCenter` + 拡大された残差をもとに戻した予測値 `trainRResPred` (python上の表記)



```python:makingData.py

elif methodModel == 2:
            # classication
            _, trainClsCenter, trainClsLoss = sess.run([trainer_cls, pred_cls_center, loss_cls], feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            # regression
            _, trainResPred, trainResLoss = sess.run([trainer_atr, res_op, loss_atr], feed_dict={y:batchY, y_label:batchlabelY})
            # alpha
            pdb.set_trace()
            _, trainAlpha, trainRResPred, trainAlphaLoss = sess.run([trainer_alpha, alpha_op, reduce_res_op, loss_alpha], feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            
            # -------------------- Test ------------------------------------- #
            if i % testPeriod == 0:
                # classication
                testClsLoss, testClsCenter = sess.run([loss_cls_test, pred_cls_center_test], feed_dict={x_cls:xTest, y:yTest, y_label:yTestlabel})    
                # regression
                testResLoss, testResPred = sess.run([loss_atr_test, res_op_test], feed_dict={y:yTest, y_label:yTestlabel})
                # test alpha
                testAlphaLoss, testAlpha, testRResPred = sess.run([loss_alpha_test, alpha_op_test, reduce_res_op_test], feed_dict={x_cls:xTest, y:yTest, y_label:yTestlabel})
                
                # Recover
                trainPred = trainClsCenter + trainRResPred
                testPred = testClsCenter + testRResPred
```

<a id="ID_3-5"></a>

## モデルの保存


```python:makingData.py
modelFileName = "model_{}_{}_{}_{}_{}_{}.ckpt".format(methodModel,sigma,nClass,pNum,nTrain,nTest)
modelPath = "models"
modelfullPath = os.path.join(modelPath,modelFileName)
saver.save(sess,modelfullPath)
```









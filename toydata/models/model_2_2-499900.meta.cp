
ů,Ô,
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
5
DivNoNan
x"T
y"T
z"T"
Ttype:
2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.1-0-g6612da8951'Ţ°1
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙2*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
p
Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙2*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
p
Placeholder_2Placeholder*
shape:˙˙˙˙˙˙˙˙˙2*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
p
Placeholder_3Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Placeholder_4Placeholder*
shape:*
dtype0*
_output_shapes
:
x
Placeholder_5Placeholder* 
shape:˙˙˙˙˙˙˙˙˙*
dtype0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

+Classify/w1/Initializer/random_normal/shapeConst*
_class
loc:@Classify/w1*
valueB"2      *
dtype0*
_output_shapes
:

*Classify/w1/Initializer/random_normal/meanConst*
_class
loc:@Classify/w1*
valueB
 *    *
dtype0*
_output_shapes
: 

,Classify/w1/Initializer/random_normal/stddevConst*
_class
loc:@Classify/w1*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ď
:Classify/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+Classify/w1/Initializer/random_normal/shape*

seed *
T0*
_class
loc:@Classify/w1*
seed2 *
dtype0*
_output_shapes
:	2
ä
)Classify/w1/Initializer/random_normal/mulMul:Classify/w1/Initializer/random_normal/RandomStandardNormal,Classify/w1/Initializer/random_normal/stddev*
T0*
_class
loc:@Classify/w1*
_output_shapes
:	2
Í
%Classify/w1/Initializer/random_normalAdd)Classify/w1/Initializer/random_normal/mul*Classify/w1/Initializer/random_normal/mean*
T0*
_class
loc:@Classify/w1*
_output_shapes
:	2
Ą
Classify/w1
VariableV2*
shared_name *
_class
loc:@Classify/w1*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ă
Classify/w1/AssignAssignClassify/w1%Classify/w1/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2
s
Classify/w1/readIdentityClassify/w1*
T0*
_class
loc:@Classify/w1*
_output_shapes
:	2

 Classify/bias1/Initializer/ConstConst*!
_class
loc:@Classify/bias1*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:

Classify/bias1
VariableV2*
shared_name *!
_class
loc:@Classify/bias1*
	container *
shape:*
dtype0*
_output_shapes	
:
Ă
Classify/bias1/AssignAssignClassify/bias1 Classify/bias1/Initializer/Const*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:
x
Classify/bias1/readIdentityClassify/bias1*
T0*!
_class
loc:@Classify/bias1*
_output_shapes	
:

Classify/MatMulMatMulPlaceholderClassify/w1/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
Classify/addAddClassify/MatMulClassify/bias1/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
Classify/ReluReluClassify/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+Classify/w2/Initializer/random_normal/shapeConst*
_class
loc:@Classify/w2*
valueB"      *
dtype0*
_output_shapes
:

*Classify/w2/Initializer/random_normal/meanConst*
_class
loc:@Classify/w2*
valueB
 *    *
dtype0*
_output_shapes
: 

,Classify/w2/Initializer/random_normal/stddevConst*
_class
loc:@Classify/w2*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
đ
:Classify/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+Classify/w2/Initializer/random_normal/shape*

seed *
T0*
_class
loc:@Classify/w2*
seed2 *
dtype0* 
_output_shapes
:

ĺ
)Classify/w2/Initializer/random_normal/mulMul:Classify/w2/Initializer/random_normal/RandomStandardNormal,Classify/w2/Initializer/random_normal/stddev*
T0*
_class
loc:@Classify/w2* 
_output_shapes
:

Î
%Classify/w2/Initializer/random_normalAdd)Classify/w2/Initializer/random_normal/mul*Classify/w2/Initializer/random_normal/mean*
T0*
_class
loc:@Classify/w2* 
_output_shapes
:

Ł
Classify/w2
VariableV2*
shared_name *
_class
loc:@Classify/w2*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ä
Classify/w2/AssignAssignClassify/w2%Classify/w2/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:

t
Classify/w2/readIdentityClassify/w2*
T0*
_class
loc:@Classify/w2* 
_output_shapes
:


 Classify/bias2/Initializer/ConstConst*!
_class
loc:@Classify/bias2*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:

Classify/bias2
VariableV2*
shared_name *!
_class
loc:@Classify/bias2*
	container *
shape:*
dtype0*
_output_shapes	
:
Ă
Classify/bias2/AssignAssignClassify/bias2 Classify/bias2/Initializer/Const*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:
x
Classify/bias2/readIdentityClassify/bias2*
T0*!
_class
loc:@Classify/bias2*
_output_shapes	
:

Classify/MatMul_1MatMulClassify/ReluClassify/w2/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Classify/add_1AddClassify/MatMul_1Classify/bias2/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
Classify/Relu_1ReluClassify/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+Classify/w3/Initializer/random_normal/shapeConst*
_class
loc:@Classify/w3*
valueB"      *
dtype0*
_output_shapes
:

*Classify/w3/Initializer/random_normal/meanConst*
_class
loc:@Classify/w3*
valueB
 *    *
dtype0*
_output_shapes
: 

,Classify/w3/Initializer/random_normal/stddevConst*
_class
loc:@Classify/w3*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
đ
:Classify/w3/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+Classify/w3/Initializer/random_normal/shape*

seed *
T0*
_class
loc:@Classify/w3*
seed2 *
dtype0* 
_output_shapes
:

ĺ
)Classify/w3/Initializer/random_normal/mulMul:Classify/w3/Initializer/random_normal/RandomStandardNormal,Classify/w3/Initializer/random_normal/stddev*
T0*
_class
loc:@Classify/w3* 
_output_shapes
:

Î
%Classify/w3/Initializer/random_normalAdd)Classify/w3/Initializer/random_normal/mul*Classify/w3/Initializer/random_normal/mean*
T0*
_class
loc:@Classify/w3* 
_output_shapes
:

Ł
Classify/w3
VariableV2*
shared_name *
_class
loc:@Classify/w3*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ä
Classify/w3/AssignAssignClassify/w3%Classify/w3/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:

t
Classify/w3/readIdentityClassify/w3*
T0*
_class
loc:@Classify/w3* 
_output_shapes
:


 Classify/bias3/Initializer/ConstConst*!
_class
loc:@Classify/bias3*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:

Classify/bias3
VariableV2*
shared_name *!
_class
loc:@Classify/bias3*
	container *
shape:*
dtype0*
_output_shapes	
:
Ă
Classify/bias3/AssignAssignClassify/bias3 Classify/bias3/Initializer/Const*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:
x
Classify/bias3/readIdentityClassify/bias3*
T0*!
_class
loc:@Classify/bias3*
_output_shapes	
:

Classify/MatMul_2MatMulClassify/Relu_1Classify/w3/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Classify/add_2AddClassify/MatMul_2Classify/bias3/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
Classify/Relu_2ReluClassify/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
-Classify/w4_1/Initializer/random_normal/shapeConst* 
_class
loc:@Classify/w4_1*
valueB"      *
dtype0*
_output_shapes
:

,Classify/w4_1/Initializer/random_normal/meanConst* 
_class
loc:@Classify/w4_1*
valueB
 *    *
dtype0*
_output_shapes
: 

.Classify/w4_1/Initializer/random_normal/stddevConst* 
_class
loc:@Classify/w4_1*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ő
<Classify/w4_1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-Classify/w4_1/Initializer/random_normal/shape*

seed *
T0* 
_class
loc:@Classify/w4_1*
seed2 *
dtype0*
_output_shapes
:	
ě
+Classify/w4_1/Initializer/random_normal/mulMul<Classify/w4_1/Initializer/random_normal/RandomStandardNormal.Classify/w4_1/Initializer/random_normal/stddev*
T0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	
Ő
'Classify/w4_1/Initializer/random_normalAdd+Classify/w4_1/Initializer/random_normal/mul,Classify/w4_1/Initializer/random_normal/mean*
T0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	
Ľ
Classify/w4_1
VariableV2*
shared_name * 
_class
loc:@Classify/w4_1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ë
Classify/w4_1/AssignAssignClassify/w4_1'Classify/w4_1/Initializer/random_normal*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	
y
Classify/w4_1/readIdentityClassify/w4_1*
T0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	

"Classify/bias4_1/Initializer/ConstConst*#
_class
loc:@Classify/bias4_1*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
Ą
Classify/bias4_1
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_1*
	container *
shape:*
dtype0*
_output_shapes
:
Ę
Classify/bias4_1/AssignAssignClassify/bias4_1"Classify/bias4_1/Initializer/Const*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:
}
Classify/bias4_1/readIdentityClassify/bias4_1*
T0*#
_class
loc:@Classify/bias4_1*
_output_shapes
:
 
-Classify/w4_2/Initializer/random_normal/shapeConst* 
_class
loc:@Classify/w4_2*
valueB"      *
dtype0*
_output_shapes
:

,Classify/w4_2/Initializer/random_normal/meanConst* 
_class
loc:@Classify/w4_2*
valueB
 *    *
dtype0*
_output_shapes
: 

.Classify/w4_2/Initializer/random_normal/stddevConst* 
_class
loc:@Classify/w4_2*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ő
<Classify/w4_2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-Classify/w4_2/Initializer/random_normal/shape*

seed *
T0* 
_class
loc:@Classify/w4_2*
seed2 *
dtype0*
_output_shapes
:	
ě
+Classify/w4_2/Initializer/random_normal/mulMul<Classify/w4_2/Initializer/random_normal/RandomStandardNormal.Classify/w4_2/Initializer/random_normal/stddev*
T0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	
Ő
'Classify/w4_2/Initializer/random_normalAdd+Classify/w4_2/Initializer/random_normal/mul,Classify/w4_2/Initializer/random_normal/mean*
T0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	
Ľ
Classify/w4_2
VariableV2*
shared_name * 
_class
loc:@Classify/w4_2*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ë
Classify/w4_2/AssignAssignClassify/w4_2'Classify/w4_2/Initializer/random_normal*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	
y
Classify/w4_2/readIdentityClassify/w4_2*
T0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	

"Classify/bias4_2/Initializer/ConstConst*#
_class
loc:@Classify/bias4_2*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
Ą
Classify/bias4_2
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_2*
	container *
shape:*
dtype0*
_output_shapes
:
Ę
Classify/bias4_2/AssignAssignClassify/bias4_2"Classify/bias4_2/Initializer/Const*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:
}
Classify/bias4_2/readIdentityClassify/bias4_2*
T0*#
_class
loc:@Classify/bias4_2*
_output_shapes
:
 
-Classify/w4_3/Initializer/random_normal/shapeConst* 
_class
loc:@Classify/w4_3*
valueB"      *
dtype0*
_output_shapes
:

,Classify/w4_3/Initializer/random_normal/meanConst* 
_class
loc:@Classify/w4_3*
valueB
 *    *
dtype0*
_output_shapes
: 

.Classify/w4_3/Initializer/random_normal/stddevConst* 
_class
loc:@Classify/w4_3*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ő
<Classify/w4_3/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-Classify/w4_3/Initializer/random_normal/shape*

seed *
T0* 
_class
loc:@Classify/w4_3*
seed2 *
dtype0*
_output_shapes
:	
ě
+Classify/w4_3/Initializer/random_normal/mulMul<Classify/w4_3/Initializer/random_normal/RandomStandardNormal.Classify/w4_3/Initializer/random_normal/stddev*
T0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	
Ő
'Classify/w4_3/Initializer/random_normalAdd+Classify/w4_3/Initializer/random_normal/mul,Classify/w4_3/Initializer/random_normal/mean*
T0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	
Ľ
Classify/w4_3
VariableV2*
shared_name * 
_class
loc:@Classify/w4_3*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ë
Classify/w4_3/AssignAssignClassify/w4_3'Classify/w4_3/Initializer/random_normal*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	
y
Classify/w4_3/readIdentityClassify/w4_3*
T0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	

"Classify/bias4_3/Initializer/ConstConst*#
_class
loc:@Classify/bias4_3*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
Ą
Classify/bias4_3
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_3*
	container *
shape:*
dtype0*
_output_shapes
:
Ę
Classify/bias4_3/AssignAssignClassify/bias4_3"Classify/bias4_3/Initializer/Const*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:
}
Classify/bias4_3/readIdentityClassify/bias4_3*
T0*#
_class
loc:@Classify/bias4_3*
_output_shapes
:

Classify/MatMul_3MatMulClassify/Relu_2Classify/w4_1/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Classify/add_3AddClassify/MatMul_3Classify/bias4_1/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify/MatMul_4MatMulClassify/Relu_2Classify/w4_2/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Classify/add_4AddClassify/MatMul_4Classify/bias4_2/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify/MatMul_5MatMulClassify/Relu_2Classify/w4_3/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Classify/add_5AddClassify/MatMul_5Classify/bias4_3/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Classify/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify/ExpandDims
ExpandDimsClassify/add_3Classify/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Classify/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify/ExpandDims_1
ExpandDimsClassify/add_4Classify/ExpandDims_1/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Classify/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify/ExpandDims_2
ExpandDimsClassify/add_5Classify/ExpandDims_2/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
Classify/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ż
Classify/concatConcatV2Classify/ExpandDimsClassify/ExpandDims_1Classify/ExpandDims_2Classify/concat/axis*

Tidx0*
T0*
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMulMatMulPlaceholderClassify/w1/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Classify_1/addAddClassify_1/MatMulClassify/bias1/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
Classify_1/ReluReluClassify_1/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMul_1MatMulClassify_1/ReluClassify/w2/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
Classify_1/add_1AddClassify_1/MatMul_1Classify/bias2/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
Classify_1/Relu_1ReluClassify_1/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMul_2MatMulClassify_1/Relu_1Classify/w3/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
Classify_1/add_2AddClassify_1/MatMul_2Classify/bias3/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
Classify_1/Relu_2ReluClassify_1/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMul_3MatMulClassify_1/Relu_2Classify/w4_1/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Classify_1/add_3AddClassify_1/MatMul_3Classify/bias4_1/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMul_4MatMulClassify_1/Relu_2Classify/w4_2/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Classify_1/add_4AddClassify_1/MatMul_4Classify/bias4_2/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Classify_1/MatMul_5MatMulClassify_1/Relu_2Classify/w4_3/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Classify_1/add_5AddClassify_1/MatMul_5Classify/bias4_3/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Classify_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify_1/ExpandDims
ExpandDimsClassify_1/add_3Classify_1/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
Classify_1/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify_1/ExpandDims_1
ExpandDimsClassify_1/add_4Classify_1/ExpandDims_1/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
Classify_1/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 

Classify_1/ExpandDims_2
ExpandDimsClassify_1/add_5Classify_1/ExpandDims_2/dim*

Tdim0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Classify_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
É
Classify_1/concatConcatV2Classify_1/ExpandDimsClassify_1/ExpandDims_1Classify_1/ExpandDims_2Classify_1/concat/axis*

Tidx0*
T0*
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
j
strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
j
strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_sliceStridedSliceClassify/concatstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
~
ArgMaxArgMaxstrided_sliceArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
CastCastArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
l

ExpandDims
ExpandDimsCastExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_1/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_1/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_1/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_1StridedSliceClassify/concatstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_1ArgMaxstrided_slice_1ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
Cast_1CastArgMax_1*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
r
ExpandDims_1
ExpandDimsCast_1ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_2/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_2/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_2/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_2StridedSliceClassify/concatstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_2ArgMaxstrided_slice_2ArgMax_2/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
Cast_2CastArgMax_2*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
r
ExpandDims_2
ExpandDimsCast_2ExpandDims_2/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
mul/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
O
mulMul
ExpandDimsmul/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
add/yConst*
valueB
 *˘N<*
dtype0*
_output_shapes
: 
H
addAddmuladd/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_1/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
U
mul_1MulExpandDims_1mul_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
add_1/yConst*
valueB
 *iqF<*
dtype0*
_output_shapes
: 
N
add_1Addmul_1add_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_2/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
U
mul_2MulExpandDims_2mul_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
add_2/yConst*
valueB
 *iqF<*
dtype0*
_output_shapes
: 
N
add_2Addmul_2add_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
y
concatConcatV2addadd_1add_2concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
subSubPlaceholder_3concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concat_1ConcatV2concatPlaceholderconcat_1/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
j
strided_slice_3/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
l
strided_slice_3/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_3/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_3StridedSliceClassify_1/concatstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_3/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_3ArgMaxstrided_slice_3ArgMax_3/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
Cast_3CastArgMax_3*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: 
r
ExpandDims_3
ExpandDimsCast_3ExpandDims_3/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_4/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_4/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_4/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_4StridedSliceClassify_1/concatstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_4/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_4ArgMaxstrided_slice_4ArgMax_4/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
Cast_4CastArgMax_4*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: 
r
ExpandDims_4
ExpandDimsCast_4ExpandDims_4/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_5/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_5/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_5/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_5StridedSliceClassify_1/concatstrided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_5/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_5ArgMaxstrided_slice_5ArgMax_5/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
Cast_5CastArgMax_5*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: 
r
ExpandDims_5
ExpandDimsCast_5ExpandDims_5/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_3/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
U
mul_3MulExpandDims_3mul_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
add_3/yConst*
valueB
 *˘N<*
dtype0*
_output_shapes
: 
N
add_3Addmul_3add_3/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_4/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
U
mul_4MulExpandDims_4mul_4/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
add_4/yConst*
valueB
 *iqF<*
dtype0*
_output_shapes
: 
N
add_4Addmul_4add_4/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
mul_5/yConst*
valueB
 *úík9*
dtype0*
_output_shapes
: 
U
mul_5MulExpandDims_5mul_5/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
add_5/yConst*
valueB
 *iqF<*
dtype0*
_output_shapes
: 
N
add_5Addmul_5add_5/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concat_2ConcatV2add_3add_4add_5concat_2/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
sub_1SubPlaceholder_3concat_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concat_3ConcatV2concat_2Placeholderconcat_3/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
˛
6ResidualRegress/w1_reg/Initializer/random_normal/shapeConst*)
_class
loc:@ResidualRegress/w1_reg*
valueB"5      *
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Initializer/random_normal/meanConst*)
_class
loc:@ResidualRegress/w1_reg*
valueB
 *    *
dtype0*
_output_shapes
: 
§
7ResidualRegress/w1_reg/Initializer/random_normal/stddevConst*)
_class
loc:@ResidualRegress/w1_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

EResidualRegress/w1_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6ResidualRegress/w1_reg/Initializer/random_normal/shape*

seed *
T0*)
_class
loc:@ResidualRegress/w1_reg*
seed2 *
dtype0*
_output_shapes
:	5

4ResidualRegress/w1_reg/Initializer/random_normal/mulMulEResidualRegress/w1_reg/Initializer/random_normal/RandomStandardNormal7ResidualRegress/w1_reg/Initializer/random_normal/stddev*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ů
0ResidualRegress/w1_reg/Initializer/random_normalAdd4ResidualRegress/w1_reg/Initializer/random_normal/mul5ResidualRegress/w1_reg/Initializer/random_normal/mean*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ˇ
ResidualRegress/w1_reg
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ď
ResidualRegress/w1_reg/AssignAssignResidualRegress/w1_reg0ResidualRegress/w1_reg/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5

ResidualRegress/w1_reg/readIdentityResidualRegress/w1_reg*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
¨
+ResidualRegress/bias1_reg/Initializer/ConstConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
ľ
ResidualRegress/bias1_reg
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ď
 ResidualRegress/bias1_reg/AssignAssignResidualRegress/bias1_reg+ResidualRegress/bias1_reg/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:

ResidualRegress/bias1_reg/readIdentityResidualRegress/bias1_reg*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
 
ResidualRegress/MatMulMatMulconcat_1ResidualRegress/w1_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress/addAddResidualRegress/MatMulResidualRegress/bias1_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
ResidualRegress/ReluReluResidualRegress/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
6ResidualRegress/w2_reg/Initializer/random_normal/shapeConst*)
_class
loc:@ResidualRegress/w2_reg*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Initializer/random_normal/meanConst*)
_class
loc:@ResidualRegress/w2_reg*
valueB
 *    *
dtype0*
_output_shapes
: 
§
7ResidualRegress/w2_reg/Initializer/random_normal/stddevConst*)
_class
loc:@ResidualRegress/w2_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

EResidualRegress/w2_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6ResidualRegress/w2_reg/Initializer/random_normal/shape*

seed *
T0*)
_class
loc:@ResidualRegress/w2_reg*
seed2 *
dtype0* 
_output_shapes
:


4ResidualRegress/w2_reg/Initializer/random_normal/mulMulEResidualRegress/w2_reg/Initializer/random_normal/RandomStandardNormal7ResidualRegress/w2_reg/Initializer/random_normal/stddev*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ú
0ResidualRegress/w2_reg/Initializer/random_normalAdd4ResidualRegress/w2_reg/Initializer/random_normal/mul5ResidualRegress/w2_reg/Initializer/random_normal/mean*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

š
ResidualRegress/w2_reg
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

đ
ResidualRegress/w2_reg/AssignAssignResidualRegress/w2_reg0ResidualRegress/w2_reg/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:


ResidualRegress/w2_reg/readIdentityResidualRegress/w2_reg*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

¨
+ResidualRegress/bias2_reg/Initializer/ConstConst*,
_class"
 loc:@ResidualRegress/bias2_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
ľ
ResidualRegress/bias2_reg
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ď
 ResidualRegress/bias2_reg/AssignAssignResidualRegress/bias2_reg+ResidualRegress/bias2_reg/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:

ResidualRegress/bias2_reg/readIdentityResidualRegress/bias2_reg*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ž
ResidualRegress/MatMul_1MatMulResidualRegress/ReluResidualRegress/w2_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress/add_1AddResidualRegress/MatMul_1ResidualRegress/bias2_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ResidualRegress/Relu_1ReluResidualRegress/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
6ResidualRegress/w3_reg/Initializer/random_normal/shapeConst*)
_class
loc:@ResidualRegress/w3_reg*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Initializer/random_normal/meanConst*)
_class
loc:@ResidualRegress/w3_reg*
valueB
 *    *
dtype0*
_output_shapes
: 
§
7ResidualRegress/w3_reg/Initializer/random_normal/stddevConst*)
_class
loc:@ResidualRegress/w3_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

EResidualRegress/w3_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6ResidualRegress/w3_reg/Initializer/random_normal/shape*

seed *
T0*)
_class
loc:@ResidualRegress/w3_reg*
seed2 *
dtype0* 
_output_shapes
:


4ResidualRegress/w3_reg/Initializer/random_normal/mulMulEResidualRegress/w3_reg/Initializer/random_normal/RandomStandardNormal7ResidualRegress/w3_reg/Initializer/random_normal/stddev*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ú
0ResidualRegress/w3_reg/Initializer/random_normalAdd4ResidualRegress/w3_reg/Initializer/random_normal/mul5ResidualRegress/w3_reg/Initializer/random_normal/mean*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

š
ResidualRegress/w3_reg
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

đ
ResidualRegress/w3_reg/AssignAssignResidualRegress/w3_reg0ResidualRegress/w3_reg/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:


ResidualRegress/w3_reg/readIdentityResidualRegress/w3_reg*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

¨
+ResidualRegress/bias3_reg/Initializer/ConstConst*,
_class"
 loc:@ResidualRegress/bias3_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
ľ
ResidualRegress/bias3_reg
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ď
 ResidualRegress/bias3_reg/AssignAssignResidualRegress/bias3_reg+ResidualRegress/bias3_reg/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:

ResidualRegress/bias3_reg/readIdentityResidualRegress/bias3_reg*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
°
ResidualRegress/MatMul_2MatMulResidualRegress/Relu_1ResidualRegress/w3_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress/add_2AddResidualRegress/MatMul_2ResidualRegress/bias3_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ResidualRegress/Relu_2ReluResidualRegress/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
6ResidualRegress/w4_reg/Initializer/random_normal/shapeConst*)
_class
loc:@ResidualRegress/w4_reg*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w4_reg/Initializer/random_normal/meanConst*)
_class
loc:@ResidualRegress/w4_reg*
valueB
 *    *
dtype0*
_output_shapes
: 
§
7ResidualRegress/w4_reg/Initializer/random_normal/stddevConst*)
_class
loc:@ResidualRegress/w4_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

EResidualRegress/w4_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6ResidualRegress/w4_reg/Initializer/random_normal/shape*

seed *
T0*)
_class
loc:@ResidualRegress/w4_reg*
seed2 *
dtype0*
_output_shapes
:	

4ResidualRegress/w4_reg/Initializer/random_normal/mulMulEResidualRegress/w4_reg/Initializer/random_normal/RandomStandardNormal7ResidualRegress/w4_reg/Initializer/random_normal/stddev*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ů
0ResidualRegress/w4_reg/Initializer/random_normalAdd4ResidualRegress/w4_reg/Initializer/random_normal/mul5ResidualRegress/w4_reg/Initializer/random_normal/mean*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ˇ
ResidualRegress/w4_reg
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ď
ResidualRegress/w4_reg/AssignAssignResidualRegress/w4_reg0ResidualRegress/w4_reg/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	

ResidualRegress/w4_reg/readIdentityResidualRegress/w4_reg*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
Ś
+ResidualRegress/bias4_reg/Initializer/ConstConst*,
_class"
 loc:@ResidualRegress/bias4_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
ł
ResidualRegress/bias4_reg
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
î
 ResidualRegress/bias4_reg/AssignAssignResidualRegress/bias4_reg+ResidualRegress/bias4_reg/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:

ResidualRegress/bias4_reg/readIdentityResidualRegress/bias4_reg*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Ż
ResidualRegress/MatMul_3MatMulResidualRegress/Relu_2ResidualRegress/w4_reg/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress/add_3AddResidualRegress/MatMul_3ResidualRegress/bias4_reg/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
ResidualRegress/SigmoidSigmoidResidualRegress/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
ResidualRegress_1/MatMulMatMulconcat_3ResidualRegress/w1_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress_1/addAddResidualRegress_1/MatMulResidualRegress/bias1_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ResidualRegress_1/ReluReluResidualRegress_1/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
ResidualRegress_1/MatMul_1MatMulResidualRegress_1/ReluResidualRegress/w2_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress_1/add_1AddResidualRegress_1/MatMul_1ResidualRegress/bias2_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
ResidualRegress_1/Relu_1ReluResidualRegress_1/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
ResidualRegress_1/MatMul_2MatMulResidualRegress_1/Relu_1ResidualRegress/w3_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress_1/add_2AddResidualRegress_1/MatMul_2ResidualRegress/bias3_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
ResidualRegress_1/Relu_2ReluResidualRegress_1/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
ResidualRegress_1/MatMul_3MatMulResidualRegress_1/Relu_2ResidualRegress/w4_reg/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ResidualRegress_1/add_3AddResidualRegress_1/MatMul_3ResidualRegress/bias4_reg/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
ResidualRegress_1/SigmoidSigmoidResidualRegress_1/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
.Regress/w1_reg/Initializer/random_normal/shapeConst*!
_class
loc:@Regress/w1_reg*
valueB"2      *
dtype0*
_output_shapes
:

-Regress/w1_reg/Initializer/random_normal/meanConst*!
_class
loc:@Regress/w1_reg*
valueB
 *    *
dtype0*
_output_shapes
: 

/Regress/w1_reg/Initializer/random_normal/stddevConst*!
_class
loc:@Regress/w1_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ř
=Regress/w1_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.Regress/w1_reg/Initializer/random_normal/shape*

seed *
T0*!
_class
loc:@Regress/w1_reg*
seed2 *
dtype0*
_output_shapes
:	2
đ
,Regress/w1_reg/Initializer/random_normal/mulMul=Regress/w1_reg/Initializer/random_normal/RandomStandardNormal/Regress/w1_reg/Initializer/random_normal/stddev*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ů
(Regress/w1_reg/Initializer/random_normalAdd,Regress/w1_reg/Initializer/random_normal/mul-Regress/w1_reg/Initializer/random_normal/mean*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
§
Regress/w1_reg
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ď
Regress/w1_reg/AssignAssignRegress/w1_reg(Regress/w1_reg/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
|
Regress/w1_reg/readIdentityRegress/w1_reg*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2

#Regress/bias1_reg/Initializer/ConstConst*$
_class
loc:@Regress/bias1_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
Ľ
Regress/bias1_reg
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ď
Regress/bias1_reg/AssignAssignRegress/bias1_reg#Regress/bias1_reg/Initializer/Const*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/readIdentityRegress/bias1_reg*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:

Regress/MatMulMatMulPlaceholder_1Regress/w1_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
Regress/addAddRegress/MatMulRegress/bias1_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Regress/ReluReluRegress/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
.Regress/w2_reg/Initializer/random_normal/shapeConst*!
_class
loc:@Regress/w2_reg*
valueB"      *
dtype0*
_output_shapes
:

-Regress/w2_reg/Initializer/random_normal/meanConst*!
_class
loc:@Regress/w2_reg*
valueB
 *    *
dtype0*
_output_shapes
: 

/Regress/w2_reg/Initializer/random_normal/stddevConst*!
_class
loc:@Regress/w2_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ů
=Regress/w2_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.Regress/w2_reg/Initializer/random_normal/shape*

seed *
T0*!
_class
loc:@Regress/w2_reg*
seed2 *
dtype0* 
_output_shapes
:

ń
,Regress/w2_reg/Initializer/random_normal/mulMul=Regress/w2_reg/Initializer/random_normal/RandomStandardNormal/Regress/w2_reg/Initializer/random_normal/stddev*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Ú
(Regress/w2_reg/Initializer/random_normalAdd,Regress/w2_reg/Initializer/random_normal/mul-Regress/w2_reg/Initializer/random_normal/mean*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Š
Regress/w2_reg
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Đ
Regress/w2_reg/AssignAssignRegress/w2_reg(Regress/w2_reg/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

}
Regress/w2_reg/readIdentityRegress/w2_reg*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:


#Regress/bias2_reg/Initializer/ConstConst*$
_class
loc:@Regress/bias2_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
Ľ
Regress/bias2_reg
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ď
Regress/bias2_reg/AssignAssignRegress/bias2_reg#Regress/bias2_reg/Initializer/Const*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/readIdentityRegress/bias2_reg*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:

Regress/MatMul_1MatMulRegress/ReluRegress/w2_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Regress/add_1AddRegress/MatMul_1Regress/bias2_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Regress/Relu_1ReluRegress/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
.Regress/w3_reg/Initializer/random_normal/shapeConst*!
_class
loc:@Regress/w3_reg*
valueB"      *
dtype0*
_output_shapes
:

-Regress/w3_reg/Initializer/random_normal/meanConst*!
_class
loc:@Regress/w3_reg*
valueB
 *    *
dtype0*
_output_shapes
: 

/Regress/w3_reg/Initializer/random_normal/stddevConst*!
_class
loc:@Regress/w3_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ů
=Regress/w3_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.Regress/w3_reg/Initializer/random_normal/shape*

seed *
T0*!
_class
loc:@Regress/w3_reg*
seed2 *
dtype0* 
_output_shapes
:

ń
,Regress/w3_reg/Initializer/random_normal/mulMul=Regress/w3_reg/Initializer/random_normal/RandomStandardNormal/Regress/w3_reg/Initializer/random_normal/stddev*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Ú
(Regress/w3_reg/Initializer/random_normalAdd,Regress/w3_reg/Initializer/random_normal/mul-Regress/w3_reg/Initializer/random_normal/mean*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Š
Regress/w3_reg
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Đ
Regress/w3_reg/AssignAssignRegress/w3_reg(Regress/w3_reg/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

}
Regress/w3_reg/readIdentityRegress/w3_reg*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:


#Regress/bias3_reg/Initializer/ConstConst*$
_class
loc:@Regress/bias3_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
Ľ
Regress/bias3_reg
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ď
Regress/bias3_reg/AssignAssignRegress/bias3_reg#Regress/bias3_reg/Initializer/Const*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/readIdentityRegress/bias3_reg*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:

Regress/MatMul_2MatMulRegress/Relu_1Regress/w3_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Regress/add_2AddRegress/MatMul_2Regress/bias3_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Regress/Relu_2ReluRegress/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
.Regress/w4_reg/Initializer/random_normal/shapeConst*!
_class
loc:@Regress/w4_reg*
valueB"      *
dtype0*
_output_shapes
:

-Regress/w4_reg/Initializer/random_normal/meanConst*!
_class
loc:@Regress/w4_reg*
valueB
 *    *
dtype0*
_output_shapes
: 

/Regress/w4_reg/Initializer/random_normal/stddevConst*!
_class
loc:@Regress/w4_reg*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ř
=Regress/w4_reg/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.Regress/w4_reg/Initializer/random_normal/shape*

seed *
T0*!
_class
loc:@Regress/w4_reg*
seed2 *
dtype0*
_output_shapes
:	
đ
,Regress/w4_reg/Initializer/random_normal/mulMul=Regress/w4_reg/Initializer/random_normal/RandomStandardNormal/Regress/w4_reg/Initializer/random_normal/stddev*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	
Ů
(Regress/w4_reg/Initializer/random_normalAdd,Regress/w4_reg/Initializer/random_normal/mul-Regress/w4_reg/Initializer/random_normal/mean*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	
§
Regress/w4_reg
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ď
Regress/w4_reg/AssignAssignRegress/w4_reg(Regress/w4_reg/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
|
Regress/w4_reg/readIdentityRegress/w4_reg*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	

#Regress/bias4_reg/Initializer/ConstConst*$
_class
loc:@Regress/bias4_reg*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
Ł
Regress/bias4_reg
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
Î
Regress/bias4_reg/AssignAssignRegress/bias4_reg#Regress/bias4_reg/Initializer/Const*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/readIdentityRegress/bias4_reg*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:

Regress/MatMul_3MatMulRegress/Relu_2Regress/w4_reg/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Regress/add_3AddRegress/MatMul_3Regress/bias4_reg/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
Regress/SigmoidSigmoidRegress/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Regress_1/MatMulMatMulPlaceholder_2Regress/w1_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
Regress_1/addAddRegress_1/MatMulRegress/bias1_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Regress_1/ReluReluRegress_1/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Regress_1/MatMul_1MatMulRegress_1/ReluRegress/w2_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Regress_1/add_1AddRegress_1/MatMul_1Regress/bias2_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Regress_1/Relu_1ReluRegress_1/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Regress_1/MatMul_2MatMulRegress_1/Relu_1Regress/w3_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Regress_1/add_2AddRegress_1/MatMul_2Regress/bias3_reg/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
Regress_1/Relu_2ReluRegress_1/add_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Regress_1/MatMul_3MatMulRegress_1/Relu_2Regress/w4_reg/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
Regress_1/add_3AddRegress_1/MatMul_3Regress/bias4_reg/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
Regress_1/SigmoidSigmoidRegress_1/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
AbsAbsResidualRegress/w1_reg/read*
T0*
_output_shapes
:	5
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
T
SumSumAbsConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
T
Abs_1AbsResidualRegress/w2_reg/read*
T0* 
_output_shapes
:

X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_1SumAbs_1Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
9
add_6AddSumSum_1*
T0*
_output_shapes
: 
S
Abs_2AbsResidualRegress/w1_reg/read*
T0*
_output_shapes
:	5
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_2SumAbs_2Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
T
Abs_3AbsResidualRegress/w2_reg/read*
T0* 
_output_shapes
:

X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_3SumAbs_3Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
;
add_7AddSum_2Sum_3*
T0*
_output_shapes
: 
K
Abs_4AbsRegress/w1_reg/read*
T0*
_output_shapes
:	2
X
Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_4SumAbs_4Const_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
Abs_5AbsRegress/w2_reg/read*
T0* 
_output_shapes
:

X
Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_5SumAbs_5Const_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
;
add_8AddSum_4Sum_5*
T0*
_output_shapes
: 
K
Abs_6AbsRegress/w1_reg/read*
T0*
_output_shapes
:	2
X
Const_6Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_6SumAbs_6Const_6*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
Abs_7AbsRegress/w2_reg/read*
T0* 
_output_shapes
:

X
Const_7Const*
valueB"       *
dtype0*
_output_shapes
:
Z
Sum_7SumAbs_7Const_7*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
;
add_9AddSum_6Sum_7*
T0*
_output_shapes
: 
N
L2LossL2LossResidualRegress/w1_reg/read*
T0*
_output_shapes
: 
P
L2Loss_1L2LossResidualRegress/w2_reg/read*
T0*
_output_shapes
: 
@
add_10AddL2LossL2Loss_1*
T0*
_output_shapes
: 
P
L2Loss_2L2LossResidualRegress/w1_reg/read*
T0*
_output_shapes
: 
P
L2Loss_3L2LossResidualRegress/w2_reg/read*
T0*
_output_shapes
: 
B
add_11AddL2Loss_2L2Loss_3*
T0*
_output_shapes
: 
H
L2Loss_4L2LossRegress/w1_reg/read*
T0*
_output_shapes
: 
H
L2Loss_5L2LossRegress/w2_reg/read*
T0*
_output_shapes
: 
B
add_12AddL2Loss_4L2Loss_5*
T0*
_output_shapes
: 
H
L2Loss_6L2LossRegress/w1_reg/read*
T0*
_output_shapes
: 
H
L2Loss_7L2LossRegress/w2_reg/read*
T0*
_output_shapes
: 
B
add_13AddL2Loss_6L2Loss_7*
T0*
_output_shapes
: 

0TrResidual/alpha/Initializer/random_normal/shapeConst*#
_class
loc:@TrResidual/alpha*
valueB:*
dtype0*
_output_shapes
:

/TrResidual/alpha/Initializer/random_normal/meanConst*#
_class
loc:@TrResidual/alpha*
valueB
 *   A*
dtype0*
_output_shapes
: 

1TrResidual/alpha/Initializer/random_normal/stddevConst*#
_class
loc:@TrResidual/alpha*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
?TrResidual/alpha/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0TrResidual/alpha/Initializer/random_normal/shape*

seed *
T0*#
_class
loc:@TrResidual/alpha*
seed2 *
dtype0*
_output_shapes
:
ó
.TrResidual/alpha/Initializer/random_normal/mulMul?TrResidual/alpha/Initializer/random_normal/RandomStandardNormal1TrResidual/alpha/Initializer/random_normal/stddev*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
:
Ü
*TrResidual/alpha/Initializer/random_normalAdd.TrResidual/alpha/Initializer/random_normal/mul/TrResidual/alpha/Initializer/random_normal/mean*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
:
Ą
TrResidual/alpha
VariableV2*
shared_name *#
_class
loc:@TrResidual/alpha*
	container *
shape:*
dtype0*
_output_shapes
:
Ň
TrResidual/alpha/AssignAssignTrResidual/alpha*TrResidual/alpha/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:
}
TrResidual/alpha/readIdentityTrResidual/alpha*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
:
^
TrResidual/MulMulTrResidual/alpha/readPlaceholder_4*
T0*
_output_shapes
:
Q
TrResidual/NegNegTrResidual/alpha/read*
T0*
_output_shapes
:
^
TrResidual/mul_1MulTrResidual/Negsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
TrResidual/ExpExpTrResidual/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
TrResidual/add/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
i
TrResidual/addAddTrResidual/add/xTrResidual/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
TrResidual/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
TrResidual/truedivRealDivTrResidual/truediv/xTrResidual/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
TrResidual_1/MulMulTrResidual/alpha/readPlaceholder_4*
T0*
_output_shapes
:
S
TrResidual_1/NegNegTrResidual/alpha/read*
T0*
_output_shapes
:
d
TrResidual_1/mul_1MulTrResidual_1/Negsub_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
TrResidual_1/ExpExpTrResidual_1/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
TrResidual_1/add/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
TrResidual_1/addAddTrResidual_1/add/xTrResidual_1/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
TrResidual_1/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
{
TrResidual_1/truedivRealDivTrResidual_1/truediv/xTrResidual_1/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
TrResidual_2/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
s
TrResidual_2/truedivRealDivTrResidual_2/truediv/xTrResidual/alpha/read*
T0*
_output_shapes
:
W
TrResidual_2/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
v
TrResidual_2/subSubTrResidual_2/sub/xResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
TrResidual_2/add/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
o
TrResidual_2/addAddTrResidual_2/subTrResidual_2/add/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
TrResidual_2/truediv_1RealDivResidualRegress/SigmoidTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
TrResidual_2/LogLogTrResidual_2/truediv_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
TrResidual_2/mulMulTrResidual_2/truedivTrResidual_2/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
TrResidual_3/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
s
TrResidual_3/truedivRealDivTrResidual_3/truediv/xTrResidual/alpha/read*
T0*
_output_shapes
:
W
TrResidual_3/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
TrResidual_3/subSubTrResidual_3/sub/xResidualRegress_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
TrResidual_3/add/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
o
TrResidual_3/addAddTrResidual_3/subTrResidual_3/add/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

TrResidual_3/truediv_1RealDivResidualRegress_1/SigmoidTrResidual_3/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
TrResidual_3/LogLogTrResidual_3/truediv_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
TrResidual_3/mulMulTrResidual_3/truedivTrResidual_3/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
add_14AddconcatTrResidual_2/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
add_15Addconcat_2TrResidual_3/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_6/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
l
strided_slice_6/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_6/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_6StridedSlicePlaceholder_5strided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_7/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
l
strided_slice_7/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_7/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_7StridedSliceClassify/concatstrided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

softmax_cross_entropy_loss/CastCaststrided_slice_6*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/softmax_cross_entropy_loss/labels_stop_gradientStopGradientsoftmax_cross_entropy_loss/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
(softmax_cross_entropy_loss/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
x
)softmax_cross_entropy_loss/xentropy/ShapeShapestrided_slice_7*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_loss/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
+softmax_cross_entropy_loss/xentropy/Shape_1Shapestrided_slice_7*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_loss/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ś
'softmax_cross_entropy_loss/xentropy/SubSub*softmax_cross_entropy_loss/xentropy/Rank_1)softmax_cross_entropy_loss/xentropy/Sub/y*
T0*
_output_shapes
: 

/softmax_cross_entropy_loss/xentropy/Slice/beginPack'softmax_cross_entropy_loss/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
x
.softmax_cross_entropy_loss/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ň
)softmax_cross_entropy_loss/xentropy/SliceSlice+softmax_cross_entropy_loss/xentropy/Shape_1/softmax_cross_entropy_loss/xentropy/Slice/begin.softmax_cross_entropy_loss/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

3softmax_cross_entropy_loss/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_loss/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

*softmax_cross_entropy_loss/xentropy/concatConcatV23softmax_cross_entropy_loss/xentropy/concat/values_0)softmax_cross_entropy_loss/xentropy/Slice/softmax_cross_entropy_loss/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
ź
+softmax_cross_entropy_loss/xentropy/ReshapeReshapestrided_slice_7*softmax_cross_entropy_loss/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

+softmax_cross_entropy_loss/xentropy/Shape_2Shape/softmax_cross_entropy_loss/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ş
)softmax_cross_entropy_loss/xentropy/Sub_1Sub*softmax_cross_entropy_loss/xentropy/Rank_2+softmax_cross_entropy_loss/xentropy/Sub_1/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss/xentropy/Slice_1/beginPack)softmax_cross_entropy_loss/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ř
+softmax_cross_entropy_loss/xentropy/Slice_1Slice+softmax_cross_entropy_loss/xentropy/Shape_21softmax_cross_entropy_loss/xentropy/Slice_1/begin0softmax_cross_entropy_loss/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss/xentropy/concat_1ConcatV25softmax_cross_entropy_loss/xentropy/concat_1/values_0+softmax_cross_entropy_loss/xentropy/Slice_11softmax_cross_entropy_loss/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ŕ
-softmax_cross_entropy_loss/xentropy/Reshape_1Reshape/softmax_cross_entropy_loss/labels_stop_gradient,softmax_cross_entropy_loss/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ę
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits+softmax_cross_entropy_loss/xentropy/Reshape-softmax_cross_entropy_loss/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
m
+softmax_cross_entropy_loss/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
¨
)softmax_cross_entropy_loss/xentropy/Sub_2Sub(softmax_cross_entropy_loss/xentropy/Rank+softmax_cross_entropy_loss/xentropy/Sub_2/y*
T0*
_output_shapes
: 
{
1softmax_cross_entropy_loss/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

0softmax_cross_entropy_loss/xentropy/Slice_2/sizePack)softmax_cross_entropy_loss/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ö
+softmax_cross_entropy_loss/xentropy/Slice_2Slice)softmax_cross_entropy_loss/xentropy/Shape1softmax_cross_entropy_loss/xentropy/Slice_2/begin0softmax_cross_entropy_loss/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ć
-softmax_cross_entropy_loss/xentropy/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy+softmax_cross_entropy_loss/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
Š
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape-softmax_cross_entropy_loss/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
ˇ
$softmax_cross_entropy_loss/ToFloat/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¨
softmax_cross_entropy_loss/MulMul-softmax_cross_entropy_loss/xentropy/Reshape_2$softmax_cross_entropy_loss/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ľ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Á
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ź
,softmax_cross_entropy_loss/num_present/EqualEqual$softmax_cross_entropy_loss/ToFloat/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
Ä
1softmax_cross_entropy_loss/num_present/zeros_likeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ç
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
É
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ű
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ë
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
ě
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
ę
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape-softmax_cross_entropy_loss/xentropy/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
é
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ż
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
ď
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape-softmax_cross_entropy_loss/xentropy/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Ç
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ó
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ł
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Š
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

 softmax_cross_entropy_loss/valueDivNoNan softmax_cross_entropy_loss/Sum_1&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
j
strided_slice_8/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_8/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_8/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_8StridedSlicePlaceholder_5strided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
strided_slice_9/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_9/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
l
strided_slice_9/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_9StridedSliceClassify/concatstrided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!softmax_cross_entropy_loss_1/CastCaststrided_slice_8*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1softmax_cross_entropy_loss_1/labels_stop_gradientStopGradient!softmax_cross_entropy_loss_1/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss_1/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
z
+softmax_cross_entropy_loss_1/xentropy/ShapeShapestrided_slice_9*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_loss_1/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
|
-softmax_cross_entropy_loss_1/xentropy/Shape_1Shapestrided_slice_9*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss_1/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)softmax_cross_entropy_loss_1/xentropy/SubSub,softmax_cross_entropy_loss_1/xentropy/Rank_1+softmax_cross_entropy_loss_1/xentropy/Sub/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss_1/xentropy/Slice/beginPack)softmax_cross_entropy_loss_1/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss_1/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ú
+softmax_cross_entropy_loss_1/xentropy/SliceSlice-softmax_cross_entropy_loss_1/xentropy/Shape_11softmax_cross_entropy_loss_1/xentropy/Slice/begin0softmax_cross_entropy_loss_1/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss_1/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss_1/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss_1/xentropy/concatConcatV25softmax_cross_entropy_loss_1/xentropy/concat/values_0+softmax_cross_entropy_loss_1/xentropy/Slice1softmax_cross_entropy_loss_1/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ŕ
-softmax_cross_entropy_loss_1/xentropy/ReshapeReshapestrided_slice_9,softmax_cross_entropy_loss_1/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
n
,softmax_cross_entropy_loss_1/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

-softmax_cross_entropy_loss_1/xentropy/Shape_2Shape1softmax_cross_entropy_loss_1/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
o
-softmax_cross_entropy_loss_1/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
°
+softmax_cross_entropy_loss_1/xentropy/Sub_1Sub,softmax_cross_entropy_loss_1/xentropy/Rank_2-softmax_cross_entropy_loss_1/xentropy/Sub_1/y*
T0*
_output_shapes
: 
˘
3softmax_cross_entropy_loss_1/xentropy/Slice_1/beginPack+softmax_cross_entropy_loss_1/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
|
2softmax_cross_entropy_loss_1/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

-softmax_cross_entropy_loss_1/xentropy/Slice_1Slice-softmax_cross_entropy_loss_1/xentropy/Shape_23softmax_cross_entropy_loss_1/xentropy/Slice_1/begin2softmax_cross_entropy_loss_1/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

7softmax_cross_entropy_loss_1/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
3softmax_cross_entropy_loss_1/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

.softmax_cross_entropy_loss_1/xentropy/concat_1ConcatV27softmax_cross_entropy_loss_1/xentropy/concat_1/values_0-softmax_cross_entropy_loss_1/xentropy/Slice_13softmax_cross_entropy_loss_1/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ć
/softmax_cross_entropy_loss_1/xentropy/Reshape_1Reshape1softmax_cross_entropy_loss_1/labels_stop_gradient.softmax_cross_entropy_loss_1/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits-softmax_cross_entropy_loss_1/xentropy/Reshape/softmax_cross_entropy_loss_1/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
o
-softmax_cross_entropy_loss_1/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
+softmax_cross_entropy_loss_1/xentropy/Sub_2Sub*softmax_cross_entropy_loss_1/xentropy/Rank-softmax_cross_entropy_loss_1/xentropy/Sub_2/y*
T0*
_output_shapes
: 
}
3softmax_cross_entropy_loss_1/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ą
2softmax_cross_entropy_loss_1/xentropy/Slice_2/sizePack+softmax_cross_entropy_loss_1/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ţ
-softmax_cross_entropy_loss_1/xentropy/Slice_2Slice+softmax_cross_entropy_loss_1/xentropy/Shape3softmax_cross_entropy_loss_1/xentropy/Slice_2/begin2softmax_cross_entropy_loss_1/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ě
/softmax_cross_entropy_loss_1/xentropy/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy-softmax_cross_entropy_loss_1/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
­
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_1/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
ť
&softmax_cross_entropy_loss_1/ToFloat/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ž
 softmax_cross_entropy_loss_1/MulMul/softmax_cross_entropy_loss_1/xentropy/Reshape_2&softmax_cross_entropy_loss_1/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ť
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
.softmax_cross_entropy_loss_1/num_present/EqualEqual&softmax_cross_entropy_loss_1/ToFloat/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
T0*
_output_shapes
: 
Č
3softmax_cross_entropy_loss_1/num_present/zeros_likeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ë
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Í
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
á
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ó
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
đ
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
î
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_1/xentropy/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
í
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Ă
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
÷
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape/softmax_cross_entropy_loss_1/xentropy/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Í
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ů
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ˇ
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Ż
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

"softmax_cross_entropy_loss_1/valueDivNoNan"softmax_cross_entropy_loss_1/Sum_1(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
t
add_16Add softmax_cross_entropy_loss/value"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
k
strided_slice_10/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_10/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_10/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_10StridedSlicePlaceholder_5strided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
strided_slice_11/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_11/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_11/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
 
strided_slice_11StridedSliceClassify/concatstrided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!softmax_cross_entropy_loss_2/CastCaststrided_slice_10*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1softmax_cross_entropy_loss_2/labels_stop_gradientStopGradient!softmax_cross_entropy_loss_2/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss_2/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
{
+softmax_cross_entropy_loss_2/xentropy/ShapeShapestrided_slice_11*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_loss_2/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
-softmax_cross_entropy_loss_2/xentropy/Shape_1Shapestrided_slice_11*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss_2/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)softmax_cross_entropy_loss_2/xentropy/SubSub,softmax_cross_entropy_loss_2/xentropy/Rank_1+softmax_cross_entropy_loss_2/xentropy/Sub/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss_2/xentropy/Slice/beginPack)softmax_cross_entropy_loss_2/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss_2/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ú
+softmax_cross_entropy_loss_2/xentropy/SliceSlice-softmax_cross_entropy_loss_2/xentropy/Shape_11softmax_cross_entropy_loss_2/xentropy/Slice/begin0softmax_cross_entropy_loss_2/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss_2/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss_2/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss_2/xentropy/concatConcatV25softmax_cross_entropy_loss_2/xentropy/concat/values_0+softmax_cross_entropy_loss_2/xentropy/Slice1softmax_cross_entropy_loss_2/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Á
-softmax_cross_entropy_loss_2/xentropy/ReshapeReshapestrided_slice_11,softmax_cross_entropy_loss_2/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
n
,softmax_cross_entropy_loss_2/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

-softmax_cross_entropy_loss_2/xentropy/Shape_2Shape1softmax_cross_entropy_loss_2/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
o
-softmax_cross_entropy_loss_2/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
°
+softmax_cross_entropy_loss_2/xentropy/Sub_1Sub,softmax_cross_entropy_loss_2/xentropy/Rank_2-softmax_cross_entropy_loss_2/xentropy/Sub_1/y*
T0*
_output_shapes
: 
˘
3softmax_cross_entropy_loss_2/xentropy/Slice_1/beginPack+softmax_cross_entropy_loss_2/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
|
2softmax_cross_entropy_loss_2/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

-softmax_cross_entropy_loss_2/xentropy/Slice_1Slice-softmax_cross_entropy_loss_2/xentropy/Shape_23softmax_cross_entropy_loss_2/xentropy/Slice_1/begin2softmax_cross_entropy_loss_2/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

7softmax_cross_entropy_loss_2/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
3softmax_cross_entropy_loss_2/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

.softmax_cross_entropy_loss_2/xentropy/concat_1ConcatV27softmax_cross_entropy_loss_2/xentropy/concat_1/values_0-softmax_cross_entropy_loss_2/xentropy/Slice_13softmax_cross_entropy_loss_2/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ć
/softmax_cross_entropy_loss_2/xentropy/Reshape_1Reshape1softmax_cross_entropy_loss_2/labels_stop_gradient.softmax_cross_entropy_loss_2/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
%softmax_cross_entropy_loss_2/xentropySoftmaxCrossEntropyWithLogits-softmax_cross_entropy_loss_2/xentropy/Reshape/softmax_cross_entropy_loss_2/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
o
-softmax_cross_entropy_loss_2/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
+softmax_cross_entropy_loss_2/xentropy/Sub_2Sub*softmax_cross_entropy_loss_2/xentropy/Rank-softmax_cross_entropy_loss_2/xentropy/Sub_2/y*
T0*
_output_shapes
: 
}
3softmax_cross_entropy_loss_2/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ą
2softmax_cross_entropy_loss_2/xentropy/Slice_2/sizePack+softmax_cross_entropy_loss_2/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ţ
-softmax_cross_entropy_loss_2/xentropy/Slice_2Slice+softmax_cross_entropy_loss_2/xentropy/Shape3softmax_cross_entropy_loss_2/xentropy/Slice_2/begin2softmax_cross_entropy_loss_2/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ě
/softmax_cross_entropy_loss_2/xentropy/Reshape_2Reshape%softmax_cross_entropy_loss_2/xentropy-softmax_cross_entropy_loss_2/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9softmax_cross_entropy_loss_2/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

?softmax_cross_entropy_loss_2/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

>softmax_cross_entropy_loss_2/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
­
>softmax_cross_entropy_loss_2/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_2/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_2/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_successNoOp
ť
&softmax_cross_entropy_loss_2/ToFloat/xConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ž
 softmax_cross_entropy_loss_2/MulMul/softmax_cross_entropy_loss_2/xentropy/Reshape_2&softmax_cross_entropy_loss_2/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"softmax_cross_entropy_loss_2/ConstConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ť
 softmax_cross_entropy_loss_2/SumSum softmax_cross_entropy_loss_2/Mul"softmax_cross_entropy_loss_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
0softmax_cross_entropy_loss_2/num_present/Equal/yConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
.softmax_cross_entropy_loss_2/num_present/EqualEqual&softmax_cross_entropy_loss_2/ToFloat/x0softmax_cross_entropy_loss_2/num_present/Equal/y*
T0*
_output_shapes
: 
Č
3softmax_cross_entropy_loss_2/num_present/zeros_likeConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ë
8softmax_cross_entropy_loss_2/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Í
8softmax_cross_entropy_loss_2/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
á
2softmax_cross_entropy_loss_2/num_present/ones_likeFill8softmax_cross_entropy_loss_2/num_present/ones_like/Shape8softmax_cross_entropy_loss_2/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ó
/softmax_cross_entropy_loss_2/num_present/SelectSelect.softmax_cross_entropy_loss_2/num_present/Equal3softmax_cross_entropy_loss_2/num_present/zeros_like2softmax_cross_entropy_loss_2/num_present/ones_like*
T0*
_output_shapes
: 
đ
]softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
î
\softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

\softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_2/xentropy/Reshape_2N^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
í
[softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Ă
ksoftmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success
÷
Jsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_like/ShapeShape/softmax_cross_entropy_loss_2/xentropy/Reshape_2N^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Í
Jsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_2/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
Dsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
:softmax_cross_entropy_loss_2/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_2/num_present/SelectDsoftmax_cross_entropy_loss_2/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
.softmax_cross_entropy_loss_2/num_present/ConstConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ů
(softmax_cross_entropy_loss_2/num_presentSum:softmax_cross_entropy_loss_2/num_present/broadcast_weights.softmax_cross_entropy_loss_2/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ˇ
$softmax_cross_entropy_loss_2/Const_1ConstN^softmax_cross_entropy_loss_2/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Ż
"softmax_cross_entropy_loss_2/Sum_1Sum softmax_cross_entropy_loss_2/Sum$softmax_cross_entropy_loss_2/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

"softmax_cross_entropy_loss_2/valueDivNoNan"softmax_cross_entropy_loss_2/Sum_1(softmax_cross_entropy_loss_2/num_present*
T0*
_output_shapes
: 
Z
add_17Addadd_16"softmax_cross_entropy_loss_2/value*
T0*
_output_shapes
: 
k
strided_slice_12/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
m
strided_slice_12/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_12/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_12StridedSlicePlaceholder_5strided_slice_12/stackstrided_slice_12/stack_1strided_slice_12/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
strided_slice_13/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
m
strided_slice_13/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_13/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
˘
strided_slice_13StridedSliceClassify_1/concatstrided_slice_13/stackstrided_slice_13/stack_1strided_slice_13/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!softmax_cross_entropy_loss_3/CastCaststrided_slice_12*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1softmax_cross_entropy_loss_3/labels_stop_gradientStopGradient!softmax_cross_entropy_loss_3/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss_3/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
{
+softmax_cross_entropy_loss_3/xentropy/ShapeShapestrided_slice_13*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_loss_3/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
-softmax_cross_entropy_loss_3/xentropy/Shape_1Shapestrided_slice_13*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss_3/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)softmax_cross_entropy_loss_3/xentropy/SubSub,softmax_cross_entropy_loss_3/xentropy/Rank_1+softmax_cross_entropy_loss_3/xentropy/Sub/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss_3/xentropy/Slice/beginPack)softmax_cross_entropy_loss_3/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss_3/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ú
+softmax_cross_entropy_loss_3/xentropy/SliceSlice-softmax_cross_entropy_loss_3/xentropy/Shape_11softmax_cross_entropy_loss_3/xentropy/Slice/begin0softmax_cross_entropy_loss_3/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss_3/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss_3/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss_3/xentropy/concatConcatV25softmax_cross_entropy_loss_3/xentropy/concat/values_0+softmax_cross_entropy_loss_3/xentropy/Slice1softmax_cross_entropy_loss_3/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Á
-softmax_cross_entropy_loss_3/xentropy/ReshapeReshapestrided_slice_13,softmax_cross_entropy_loss_3/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
n
,softmax_cross_entropy_loss_3/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

-softmax_cross_entropy_loss_3/xentropy/Shape_2Shape1softmax_cross_entropy_loss_3/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
o
-softmax_cross_entropy_loss_3/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
°
+softmax_cross_entropy_loss_3/xentropy/Sub_1Sub,softmax_cross_entropy_loss_3/xentropy/Rank_2-softmax_cross_entropy_loss_3/xentropy/Sub_1/y*
T0*
_output_shapes
: 
˘
3softmax_cross_entropy_loss_3/xentropy/Slice_1/beginPack+softmax_cross_entropy_loss_3/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
|
2softmax_cross_entropy_loss_3/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

-softmax_cross_entropy_loss_3/xentropy/Slice_1Slice-softmax_cross_entropy_loss_3/xentropy/Shape_23softmax_cross_entropy_loss_3/xentropy/Slice_1/begin2softmax_cross_entropy_loss_3/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

7softmax_cross_entropy_loss_3/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
3softmax_cross_entropy_loss_3/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

.softmax_cross_entropy_loss_3/xentropy/concat_1ConcatV27softmax_cross_entropy_loss_3/xentropy/concat_1/values_0-softmax_cross_entropy_loss_3/xentropy/Slice_13softmax_cross_entropy_loss_3/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ć
/softmax_cross_entropy_loss_3/xentropy/Reshape_1Reshape1softmax_cross_entropy_loss_3/labels_stop_gradient.softmax_cross_entropy_loss_3/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
%softmax_cross_entropy_loss_3/xentropySoftmaxCrossEntropyWithLogits-softmax_cross_entropy_loss_3/xentropy/Reshape/softmax_cross_entropy_loss_3/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
o
-softmax_cross_entropy_loss_3/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
+softmax_cross_entropy_loss_3/xentropy/Sub_2Sub*softmax_cross_entropy_loss_3/xentropy/Rank-softmax_cross_entropy_loss_3/xentropy/Sub_2/y*
T0*
_output_shapes
: 
}
3softmax_cross_entropy_loss_3/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ą
2softmax_cross_entropy_loss_3/xentropy/Slice_2/sizePack+softmax_cross_entropy_loss_3/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ţ
-softmax_cross_entropy_loss_3/xentropy/Slice_2Slice+softmax_cross_entropy_loss_3/xentropy/Shape3softmax_cross_entropy_loss_3/xentropy/Slice_2/begin2softmax_cross_entropy_loss_3/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ě
/softmax_cross_entropy_loss_3/xentropy/Reshape_2Reshape%softmax_cross_entropy_loss_3/xentropy-softmax_cross_entropy_loss_3/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9softmax_cross_entropy_loss_3/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

?softmax_cross_entropy_loss_3/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

>softmax_cross_entropy_loss_3/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
­
>softmax_cross_entropy_loss_3/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_3/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_3/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_successNoOp
ť
&softmax_cross_entropy_loss_3/ToFloat/xConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ž
 softmax_cross_entropy_loss_3/MulMul/softmax_cross_entropy_loss_3/xentropy/Reshape_2&softmax_cross_entropy_loss_3/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"softmax_cross_entropy_loss_3/ConstConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ť
 softmax_cross_entropy_loss_3/SumSum softmax_cross_entropy_loss_3/Mul"softmax_cross_entropy_loss_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
0softmax_cross_entropy_loss_3/num_present/Equal/yConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
.softmax_cross_entropy_loss_3/num_present/EqualEqual&softmax_cross_entropy_loss_3/ToFloat/x0softmax_cross_entropy_loss_3/num_present/Equal/y*
T0*
_output_shapes
: 
Č
3softmax_cross_entropy_loss_3/num_present/zeros_likeConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ë
8softmax_cross_entropy_loss_3/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Í
8softmax_cross_entropy_loss_3/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
á
2softmax_cross_entropy_loss_3/num_present/ones_likeFill8softmax_cross_entropy_loss_3/num_present/ones_like/Shape8softmax_cross_entropy_loss_3/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ó
/softmax_cross_entropy_loss_3/num_present/SelectSelect.softmax_cross_entropy_loss_3/num_present/Equal3softmax_cross_entropy_loss_3/num_present/zeros_like2softmax_cross_entropy_loss_3/num_present/ones_like*
T0*
_output_shapes
: 
đ
]softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
î
\softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

\softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_3/xentropy/Reshape_2N^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
í
[softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Ă
ksoftmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success
÷
Jsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_like/ShapeShape/softmax_cross_entropy_loss_3/xentropy/Reshape_2N^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Í
Jsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_3/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
Dsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
:softmax_cross_entropy_loss_3/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_3/num_present/SelectDsoftmax_cross_entropy_loss_3/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
.softmax_cross_entropy_loss_3/num_present/ConstConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ů
(softmax_cross_entropy_loss_3/num_presentSum:softmax_cross_entropy_loss_3/num_present/broadcast_weights.softmax_cross_entropy_loss_3/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ˇ
$softmax_cross_entropy_loss_3/Const_1ConstN^softmax_cross_entropy_loss_3/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Ż
"softmax_cross_entropy_loss_3/Sum_1Sum softmax_cross_entropy_loss_3/Sum$softmax_cross_entropy_loss_3/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

"softmax_cross_entropy_loss_3/valueDivNoNan"softmax_cross_entropy_loss_3/Sum_1(softmax_cross_entropy_loss_3/num_present*
T0*
_output_shapes
: 
k
strided_slice_14/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_14/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_14/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_14StridedSlicePlaceholder_5strided_slice_14/stackstrided_slice_14/stack_1strided_slice_14/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
strided_slice_15/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_15/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_15/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
˘
strided_slice_15StridedSliceClassify_1/concatstrided_slice_15/stackstrided_slice_15/stack_1strided_slice_15/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!softmax_cross_entropy_loss_4/CastCaststrided_slice_14*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1softmax_cross_entropy_loss_4/labels_stop_gradientStopGradient!softmax_cross_entropy_loss_4/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss_4/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
{
+softmax_cross_entropy_loss_4/xentropy/ShapeShapestrided_slice_15*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_loss_4/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
-softmax_cross_entropy_loss_4/xentropy/Shape_1Shapestrided_slice_15*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss_4/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)softmax_cross_entropy_loss_4/xentropy/SubSub,softmax_cross_entropy_loss_4/xentropy/Rank_1+softmax_cross_entropy_loss_4/xentropy/Sub/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss_4/xentropy/Slice/beginPack)softmax_cross_entropy_loss_4/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss_4/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ú
+softmax_cross_entropy_loss_4/xentropy/SliceSlice-softmax_cross_entropy_loss_4/xentropy/Shape_11softmax_cross_entropy_loss_4/xentropy/Slice/begin0softmax_cross_entropy_loss_4/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss_4/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss_4/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss_4/xentropy/concatConcatV25softmax_cross_entropy_loss_4/xentropy/concat/values_0+softmax_cross_entropy_loss_4/xentropy/Slice1softmax_cross_entropy_loss_4/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Á
-softmax_cross_entropy_loss_4/xentropy/ReshapeReshapestrided_slice_15,softmax_cross_entropy_loss_4/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
n
,softmax_cross_entropy_loss_4/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

-softmax_cross_entropy_loss_4/xentropy/Shape_2Shape1softmax_cross_entropy_loss_4/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
o
-softmax_cross_entropy_loss_4/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
°
+softmax_cross_entropy_loss_4/xentropy/Sub_1Sub,softmax_cross_entropy_loss_4/xentropy/Rank_2-softmax_cross_entropy_loss_4/xentropy/Sub_1/y*
T0*
_output_shapes
: 
˘
3softmax_cross_entropy_loss_4/xentropy/Slice_1/beginPack+softmax_cross_entropy_loss_4/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
|
2softmax_cross_entropy_loss_4/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

-softmax_cross_entropy_loss_4/xentropy/Slice_1Slice-softmax_cross_entropy_loss_4/xentropy/Shape_23softmax_cross_entropy_loss_4/xentropy/Slice_1/begin2softmax_cross_entropy_loss_4/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

7softmax_cross_entropy_loss_4/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
3softmax_cross_entropy_loss_4/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

.softmax_cross_entropy_loss_4/xentropy/concat_1ConcatV27softmax_cross_entropy_loss_4/xentropy/concat_1/values_0-softmax_cross_entropy_loss_4/xentropy/Slice_13softmax_cross_entropy_loss_4/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ć
/softmax_cross_entropy_loss_4/xentropy/Reshape_1Reshape1softmax_cross_entropy_loss_4/labels_stop_gradient.softmax_cross_entropy_loss_4/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
%softmax_cross_entropy_loss_4/xentropySoftmaxCrossEntropyWithLogits-softmax_cross_entropy_loss_4/xentropy/Reshape/softmax_cross_entropy_loss_4/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
o
-softmax_cross_entropy_loss_4/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
+softmax_cross_entropy_loss_4/xentropy/Sub_2Sub*softmax_cross_entropy_loss_4/xentropy/Rank-softmax_cross_entropy_loss_4/xentropy/Sub_2/y*
T0*
_output_shapes
: 
}
3softmax_cross_entropy_loss_4/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ą
2softmax_cross_entropy_loss_4/xentropy/Slice_2/sizePack+softmax_cross_entropy_loss_4/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ţ
-softmax_cross_entropy_loss_4/xentropy/Slice_2Slice+softmax_cross_entropy_loss_4/xentropy/Shape3softmax_cross_entropy_loss_4/xentropy/Slice_2/begin2softmax_cross_entropy_loss_4/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ě
/softmax_cross_entropy_loss_4/xentropy/Reshape_2Reshape%softmax_cross_entropy_loss_4/xentropy-softmax_cross_entropy_loss_4/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9softmax_cross_entropy_loss_4/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

?softmax_cross_entropy_loss_4/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

>softmax_cross_entropy_loss_4/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
­
>softmax_cross_entropy_loss_4/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_4/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_4/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_successNoOp
ť
&softmax_cross_entropy_loss_4/ToFloat/xConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ž
 softmax_cross_entropy_loss_4/MulMul/softmax_cross_entropy_loss_4/xentropy/Reshape_2&softmax_cross_entropy_loss_4/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"softmax_cross_entropy_loss_4/ConstConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ť
 softmax_cross_entropy_loss_4/SumSum softmax_cross_entropy_loss_4/Mul"softmax_cross_entropy_loss_4/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
0softmax_cross_entropy_loss_4/num_present/Equal/yConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
.softmax_cross_entropy_loss_4/num_present/EqualEqual&softmax_cross_entropy_loss_4/ToFloat/x0softmax_cross_entropy_loss_4/num_present/Equal/y*
T0*
_output_shapes
: 
Č
3softmax_cross_entropy_loss_4/num_present/zeros_likeConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ë
8softmax_cross_entropy_loss_4/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Í
8softmax_cross_entropy_loss_4/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
á
2softmax_cross_entropy_loss_4/num_present/ones_likeFill8softmax_cross_entropy_loss_4/num_present/ones_like/Shape8softmax_cross_entropy_loss_4/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ó
/softmax_cross_entropy_loss_4/num_present/SelectSelect.softmax_cross_entropy_loss_4/num_present/Equal3softmax_cross_entropy_loss_4/num_present/zeros_like2softmax_cross_entropy_loss_4/num_present/ones_like*
T0*
_output_shapes
: 
đ
]softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
î
\softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

\softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_4/xentropy/Reshape_2N^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
í
[softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Ă
ksoftmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success
÷
Jsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_like/ShapeShape/softmax_cross_entropy_loss_4/xentropy/Reshape_2N^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Í
Jsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_4/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
Dsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
:softmax_cross_entropy_loss_4/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_4/num_present/SelectDsoftmax_cross_entropy_loss_4/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
.softmax_cross_entropy_loss_4/num_present/ConstConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ů
(softmax_cross_entropy_loss_4/num_presentSum:softmax_cross_entropy_loss_4/num_present/broadcast_weights.softmax_cross_entropy_loss_4/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ˇ
$softmax_cross_entropy_loss_4/Const_1ConstN^softmax_cross_entropy_loss_4/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Ż
"softmax_cross_entropy_loss_4/Sum_1Sum softmax_cross_entropy_loss_4/Sum$softmax_cross_entropy_loss_4/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

"softmax_cross_entropy_loss_4/valueDivNoNan"softmax_cross_entropy_loss_4/Sum_1(softmax_cross_entropy_loss_4/num_present*
T0*
_output_shapes
: 
v
add_18Add"softmax_cross_entropy_loss_3/value"softmax_cross_entropy_loss_4/value*
T0*
_output_shapes
: 
k
strided_slice_16/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_16/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_16/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:

strided_slice_16StridedSlicePlaceholder_5strided_slice_16/stackstrided_slice_16/stack_1strided_slice_16/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
strided_slice_17/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_17/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
m
strided_slice_17/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
˘
strided_slice_17StridedSliceClassify_1/concatstrided_slice_17/stackstrided_slice_17/stack_1strided_slice_17/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!softmax_cross_entropy_loss_5/CastCaststrided_slice_16*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1softmax_cross_entropy_loss_5/labels_stop_gradientStopGradient!softmax_cross_entropy_loss_5/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
*softmax_cross_entropy_loss_5/xentropy/RankConst*
value	B :*
dtype0*
_output_shapes
: 
{
+softmax_cross_entropy_loss_5/xentropy/ShapeShapestrided_slice_17*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_loss_5/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
}
-softmax_cross_entropy_loss_5/xentropy/Shape_1Shapestrided_slice_17*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_loss_5/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ź
)softmax_cross_entropy_loss_5/xentropy/SubSub,softmax_cross_entropy_loss_5/xentropy/Rank_1+softmax_cross_entropy_loss_5/xentropy/Sub/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss_5/xentropy/Slice/beginPack)softmax_cross_entropy_loss_5/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss_5/xentropy/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ú
+softmax_cross_entropy_loss_5/xentropy/SliceSlice-softmax_cross_entropy_loss_5/xentropy/Shape_11softmax_cross_entropy_loss_5/xentropy/Slice/begin0softmax_cross_entropy_loss_5/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:

5softmax_cross_entropy_loss_5/xentropy/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss_5/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

,softmax_cross_entropy_loss_5/xentropy/concatConcatV25softmax_cross_entropy_loss_5/xentropy/concat/values_0+softmax_cross_entropy_loss_5/xentropy/Slice1softmax_cross_entropy_loss_5/xentropy/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Á
-softmax_cross_entropy_loss_5/xentropy/ReshapeReshapestrided_slice_17,softmax_cross_entropy_loss_5/xentropy/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
n
,softmax_cross_entropy_loss_5/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

-softmax_cross_entropy_loss_5/xentropy/Shape_2Shape1softmax_cross_entropy_loss_5/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
o
-softmax_cross_entropy_loss_5/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
°
+softmax_cross_entropy_loss_5/xentropy/Sub_1Sub,softmax_cross_entropy_loss_5/xentropy/Rank_2-softmax_cross_entropy_loss_5/xentropy/Sub_1/y*
T0*
_output_shapes
: 
˘
3softmax_cross_entropy_loss_5/xentropy/Slice_1/beginPack+softmax_cross_entropy_loss_5/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
|
2softmax_cross_entropy_loss_5/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

-softmax_cross_entropy_loss_5/xentropy/Slice_1Slice-softmax_cross_entropy_loss_5/xentropy/Shape_23softmax_cross_entropy_loss_5/xentropy/Slice_1/begin2softmax_cross_entropy_loss_5/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:

7softmax_cross_entropy_loss_5/xentropy/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
u
3softmax_cross_entropy_loss_5/xentropy/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

.softmax_cross_entropy_loss_5/xentropy/concat_1ConcatV27softmax_cross_entropy_loss_5/xentropy/concat_1/values_0-softmax_cross_entropy_loss_5/xentropy/Slice_13softmax_cross_entropy_loss_5/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ć
/softmax_cross_entropy_loss_5/xentropy/Reshape_1Reshape1softmax_cross_entropy_loss_5/labels_stop_gradient.softmax_cross_entropy_loss_5/xentropy/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ
%softmax_cross_entropy_loss_5/xentropySoftmaxCrossEntropyWithLogits-softmax_cross_entropy_loss_5/xentropy/Reshape/softmax_cross_entropy_loss_5/xentropy/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
o
-softmax_cross_entropy_loss_5/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
+softmax_cross_entropy_loss_5/xentropy/Sub_2Sub*softmax_cross_entropy_loss_5/xentropy/Rank-softmax_cross_entropy_loss_5/xentropy/Sub_2/y*
T0*
_output_shapes
: 
}
3softmax_cross_entropy_loss_5/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ą
2softmax_cross_entropy_loss_5/xentropy/Slice_2/sizePack+softmax_cross_entropy_loss_5/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
ţ
-softmax_cross_entropy_loss_5/xentropy/Slice_2Slice+softmax_cross_entropy_loss_5/xentropy/Shape3softmax_cross_entropy_loss_5/xentropy/Slice_2/begin2softmax_cross_entropy_loss_5/xentropy/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ě
/softmax_cross_entropy_loss_5/xentropy/Reshape_2Reshape%softmax_cross_entropy_loss_5/xentropy-softmax_cross_entropy_loss_5/xentropy/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9softmax_cross_entropy_loss_5/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

?softmax_cross_entropy_loss_5/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

>softmax_cross_entropy_loss_5/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
­
>softmax_cross_entropy_loss_5/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_5/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_5/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
U
Msoftmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_successNoOp
ť
&softmax_cross_entropy_loss_5/ToFloat/xConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ž
 softmax_cross_entropy_loss_5/MulMul/softmax_cross_entropy_loss_5/xentropy/Reshape_2&softmax_cross_entropy_loss_5/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"softmax_cross_entropy_loss_5/ConstConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ť
 softmax_cross_entropy_loss_5/SumSum softmax_cross_entropy_loss_5/Mul"softmax_cross_entropy_loss_5/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
0softmax_cross_entropy_loss_5/num_present/Equal/yConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
.softmax_cross_entropy_loss_5/num_present/EqualEqual&softmax_cross_entropy_loss_5/ToFloat/x0softmax_cross_entropy_loss_5/num_present/Equal/y*
T0*
_output_shapes
: 
Č
3softmax_cross_entropy_loss_5/num_present/zeros_likeConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ë
8softmax_cross_entropy_loss_5/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Í
8softmax_cross_entropy_loss_5/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
á
2softmax_cross_entropy_loss_5/num_present/ones_likeFill8softmax_cross_entropy_loss_5/num_present/ones_like/Shape8softmax_cross_entropy_loss_5/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
ó
/softmax_cross_entropy_loss_5/num_present/SelectSelect.softmax_cross_entropy_loss_5/num_present/Equal3softmax_cross_entropy_loss_5/num_present/zeros_like2softmax_cross_entropy_loss_5/num_present/ones_like*
T0*
_output_shapes
: 
đ
]softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
î
\softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

\softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/values/shapeShape/softmax_cross_entropy_loss_5/xentropy/Reshape_2N^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
í
[softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Ă
ksoftmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success
÷
Jsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_like/ShapeShape/softmax_cross_entropy_loss_5/xentropy/Reshape_2N^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Í
Jsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_5/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
Dsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
:softmax_cross_entropy_loss_5/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_5/num_present/SelectDsoftmax_cross_entropy_loss_5/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
.softmax_cross_entropy_loss_5/num_present/ConstConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
Ů
(softmax_cross_entropy_loss_5/num_presentSum:softmax_cross_entropy_loss_5/num_present/broadcast_weights.softmax_cross_entropy_loss_5/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ˇ
$softmax_cross_entropy_loss_5/Const_1ConstN^softmax_cross_entropy_loss_5/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Ż
"softmax_cross_entropy_loss_5/Sum_1Sum softmax_cross_entropy_loss_5/Sum$softmax_cross_entropy_loss_5/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

"softmax_cross_entropy_loss_5/valueDivNoNan"softmax_cross_entropy_loss_5/Sum_1(softmax_cross_entropy_loss_5/num_present*
T0*
_output_shapes
: 
Z
add_19Addadd_18"softmax_cross_entropy_loss_5/value*
T0*
_output_shapes
: 
^
sub_2SubPlaceholder_3Regress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
I
SquareSquaresub_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Const_8Const*
valueB"       *
dtype0*
_output_shapes
:
[
MeanMeanSquareConst_8*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
`
sub_3SubPlaceholder_3Regress_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_1Squaresub_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Const_9Const*
valueB"       *
dtype0*
_output_shapes
:
_
Mean_1MeanSquare_1Const_9*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
;
add_20AddMeanadd_8*
T0*
_output_shapes
: 
=
add_21AddMean_1add_9*
T0*
_output_shapes
: 
<
add_22AddMeanadd_12*
T0*
_output_shapes
: 
>
add_23AddMean_1add_13*
T0*
_output_shapes
: 
\
sub_4SubsubResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_2Squaresub_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Const_10Const*
valueB"       *
dtype0*
_output_shapes
:
`
Mean_2MeanSquare_2Const_10*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
`
sub_5Subsub_1ResidualRegress_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_3Squaresub_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Const_11Const*
valueB"       *
dtype0*
_output_shapes
:
`
Mean_3MeanSquare_3Const_11*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
=
add_24AddMean_2add_6*
T0*
_output_shapes
: 
=
add_25AddMean_3add_7*
T0*
_output_shapes
: 
>
add_26AddMean_2add_10*
T0*
_output_shapes
: 
>
add_27AddMean_3add_11*
T0*
_output_shapes
: 
k
sub_6SubTrResidual/truedivResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_4Squaresub_6*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Const_12Const*
valueB"       *
dtype0*
_output_shapes
:
`
Mean_4MeanSquare_4Const_12*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
o
sub_7SubTrResidual_1/truedivResidualRegress_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_5Squaresub_7*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Const_13Const*
valueB"       *
dtype0*
_output_shapes
:
`
Mean_5MeanSquare_5Const_13*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
=
add_28AddMean_4add_6*
T0*
_output_shapes
: 
=
add_29AddMean_5add_7*
T0*
_output_shapes
: 
>
add_30AddMean_4add_10*
T0*
_output_shapes
: 
>
add_31AddMean_5add_11*
T0*
_output_shapes
: 
U
gradients/ShapeShapeadd_14*
T0*
out_type0*
_output_shapes
:
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/add_14_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
m
gradients/add_14_grad/Shape_1ShapeTrResidual_2/mul*
T0*
out_type0*
_output_shapes
:
˝
+gradients/add_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_14_grad/Shapegradients/add_14_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/add_14_grad/SumSumgradients/Fill+gradients/add_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
 
gradients/add_14_grad/ReshapeReshapegradients/add_14_grad/Sumgradients/add_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/add_14_grad/Sum_1Sumgradients/Fill-gradients/add_14_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ś
gradients/add_14_grad/Reshape_1Reshapegradients/add_14_grad/Sum_1gradients/add_14_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients/TrResidual_2/mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
w
'gradients/TrResidual_2/mul_grad/Shape_1ShapeTrResidual_2/Log*
T0*
out_type0*
_output_shapes
:
Ű
5gradients/TrResidual_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/TrResidual_2/mul_grad/Shape'gradients/TrResidual_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

#gradients/TrResidual_2/mul_grad/MulMulgradients/add_14_grad/Reshape_1TrResidual_2/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
#gradients/TrResidual_2/mul_grad/SumSum#gradients/TrResidual_2/mul_grad/Mul5gradients/TrResidual_2/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ą
'gradients/TrResidual_2/mul_grad/ReshapeReshape#gradients/TrResidual_2/mul_grad/Sum%gradients/TrResidual_2/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

%gradients/TrResidual_2/mul_grad/Mul_1MulTrResidual_2/truedivgradients/add_14_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
%gradients/TrResidual_2/mul_grad/Sum_1Sum%gradients/TrResidual_2/mul_grad/Mul_17gradients/TrResidual_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ä
)gradients/TrResidual_2/mul_grad/Reshape_1Reshape%gradients/TrResidual_2/mul_grad/Sum_1'gradients/TrResidual_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
*gradients/TrResidual_2/Log_grad/Reciprocal
ReciprocalTrResidual_2/truediv_1*^gradients/TrResidual_2/mul_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
#gradients/TrResidual_2/Log_grad/mulMul)gradients/TrResidual_2/mul_grad/Reshape_1*gradients/TrResidual_2/Log_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/TrResidual_2/truediv_1_grad/ShapeShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
}
-gradients/TrResidual_2/truediv_1_grad/Shape_1ShapeTrResidual_2/add*
T0*
out_type0*
_output_shapes
:
í
;gradients/TrResidual_2/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/TrResidual_2/truediv_1_grad/Shape-gradients/TrResidual_2/truediv_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
-gradients/TrResidual_2/truediv_1_grad/RealDivRealDiv#gradients/TrResidual_2/Log_grad/mulTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
)gradients/TrResidual_2/truediv_1_grad/SumSum-gradients/TrResidual_2/truediv_1_grad/RealDiv;gradients/TrResidual_2/truediv_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Đ
-gradients/TrResidual_2/truediv_1_grad/ReshapeReshape)gradients/TrResidual_2/truediv_1_grad/Sum+gradients/TrResidual_2/truediv_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
)gradients/TrResidual_2/truediv_1_grad/NegNegResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
/gradients/TrResidual_2/truediv_1_grad/RealDiv_1RealDiv)gradients/TrResidual_2/truediv_1_grad/NegTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
/gradients/TrResidual_2/truediv_1_grad/RealDiv_2RealDiv/gradients/TrResidual_2/truediv_1_grad/RealDiv_1TrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
)gradients/TrResidual_2/truediv_1_grad/mulMul#gradients/TrResidual_2/Log_grad/mul/gradients/TrResidual_2/truediv_1_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
+gradients/TrResidual_2/truediv_1_grad/Sum_1Sum)gradients/TrResidual_2/truediv_1_grad/mul=gradients/TrResidual_2/truediv_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ö
/gradients/TrResidual_2/truediv_1_grad/Reshape_1Reshape+gradients/TrResidual_2/truediv_1_grad/Sum_1-gradients/TrResidual_2/truediv_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
%gradients/TrResidual_2/add_grad/ShapeShapeTrResidual_2/sub*
T0*
out_type0*
_output_shapes
:
j
'gradients/TrResidual_2/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ű
5gradients/TrResidual_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/TrResidual_2/add_grad/Shape'gradients/TrResidual_2/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ň
#gradients/TrResidual_2/add_grad/SumSum/gradients/TrResidual_2/truediv_1_grad/Reshape_15gradients/TrResidual_2/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
'gradients/TrResidual_2/add_grad/ReshapeReshape#gradients/TrResidual_2/add_grad/Sum%gradients/TrResidual_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
%gradients/TrResidual_2/add_grad/Sum_1Sum/gradients/TrResidual_2/truediv_1_grad/Reshape_17gradients/TrResidual_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ł
)gradients/TrResidual_2/add_grad/Reshape_1Reshape%gradients/TrResidual_2/add_grad/Sum_1'gradients/TrResidual_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
h
%gradients/TrResidual_2/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
~
'gradients/TrResidual_2/sub_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ű
5gradients/TrResidual_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/TrResidual_2/sub_grad/Shape'gradients/TrResidual_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
#gradients/TrResidual_2/sub_grad/SumSum'gradients/TrResidual_2/add_grad/Reshape5gradients/TrResidual_2/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
­
'gradients/TrResidual_2/sub_grad/ReshapeReshape#gradients/TrResidual_2/sub_grad/Sum%gradients/TrResidual_2/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Î
%gradients/TrResidual_2/sub_grad/Sum_1Sum'gradients/TrResidual_2/add_grad/Reshape7gradients/TrResidual_2/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
t
#gradients/TrResidual_2/sub_grad/NegNeg%gradients/TrResidual_2/sub_grad/Sum_1*
T0*
_output_shapes
:
Â
)gradients/TrResidual_2/sub_grad/Reshape_1Reshape#gradients/TrResidual_2/sub_grad/Neg'gradients/TrResidual_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
gradients/AddNAddN-gradients/TrResidual_2/truediv_1_grad/Reshape)gradients/TrResidual_2/sub_grad/Reshape_1*
T0*@
_class6
42loc:@gradients/TrResidual_2/truediv_1_grad/Reshape*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2gradients/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoidgradients/AddN*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
v
,gradients/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/ResidualRegress/add_3_grad/Shape,gradients/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ß
(gradients/ResidualRegress/add_3_grad/SumSum2gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad:gradients/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
,gradients/ResidualRegress/add_3_grad/ReshapeReshape(gradients/ResidualRegress/add_3_grad/Sum*gradients/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
*gradients/ResidualRegress/add_3_grad/Sum_1Sum2gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ć
.gradients/ResidualRegress/add_3_grad/Reshape_1Reshape*gradients/ResidualRegress/add_3_grad/Sum_1,gradients/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ü
.gradients/ResidualRegress/MatMul_3_grad/MatMulMatMul,gradients/ResidualRegress/add_3_grad/ReshapeResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
0gradients/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2,gradients/ResidualRegress/add_3_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
ľ
.gradients/ResidualRegress/Relu_2_grad/ReluGradReluGrad.gradients/ResidualRegress/MatMul_3_grad/MatMulResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
w
,gradients/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/ResidualRegress/add_2_grad/Shape,gradients/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients/ResidualRegress/add_2_grad/SumSum.gradients/ResidualRegress/Relu_2_grad/ReluGrad:gradients/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients/ResidualRegress/add_2_grad/ReshapeReshape(gradients/ResidualRegress/add_2_grad/Sum*gradients/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients/ResidualRegress/add_2_grad/Sum_1Sum.gradients/ResidualRegress/Relu_2_grad/ReluGrad<gradients/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients/ResidualRegress/add_2_grad/Reshape_1Reshape*gradients/ResidualRegress/add_2_grad/Sum_1,gradients/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ü
.gradients/ResidualRegress/MatMul_2_grad/MatMulMatMul,gradients/ResidualRegress/add_2_grad/ReshapeResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
0gradients/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1,gradients/ResidualRegress/add_2_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

ľ
.gradients/ResidualRegress/Relu_1_grad/ReluGradReluGrad.gradients/ResidualRegress/MatMul_2_grad/MatMulResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
w
,gradients/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/ResidualRegress/add_1_grad/Shape,gradients/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients/ResidualRegress/add_1_grad/SumSum.gradients/ResidualRegress/Relu_1_grad/ReluGrad:gradients/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients/ResidualRegress/add_1_grad/ReshapeReshape(gradients/ResidualRegress/add_1_grad/Sum*gradients/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients/ResidualRegress/add_1_grad/Sum_1Sum.gradients/ResidualRegress/Relu_1_grad/ReluGrad<gradients/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients/ResidualRegress/add_1_grad/Reshape_1Reshape*gradients/ResidualRegress/add_1_grad/Sum_1,gradients/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ü
.gradients/ResidualRegress/MatMul_1_grad/MatMulMatMul,gradients/ResidualRegress/add_1_grad/ReshapeResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
0gradients/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu,gradients/ResidualRegress/add_1_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

ą
,gradients/ResidualRegress/Relu_grad/ReluGradReluGrad.gradients/ResidualRegress/MatMul_1_grad/MatMulResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
(gradients/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
u
*gradients/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ä
8gradients/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/ResidualRegress/add_grad/Shape*gradients/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ő
&gradients/ResidualRegress/add_grad/SumSum,gradients/ResidualRegress/Relu_grad/ReluGrad8gradients/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Č
*gradients/ResidualRegress/add_grad/ReshapeReshape&gradients/ResidualRegress/add_grad/Sum(gradients/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
(gradients/ResidualRegress/add_grad/Sum_1Sum,gradients/ResidualRegress/Relu_grad/ReluGrad:gradients/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Á
,gradients/ResidualRegress/add_grad/Reshape_1Reshape(gradients/ResidualRegress/add_grad/Sum_1*gradients/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
×
,gradients/ResidualRegress/MatMul_grad/MatMulMatMul*gradients/ResidualRegress/add_grad/ReshapeResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
ž
.gradients/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1*gradients/ResidualRegress/add_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
T0*
_output_shapes
: 
c
gradients/concat_1_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:

gradients/concat_1_grad/ShapeNShapeNconcatPlaceholder*
T0*
out_type0*
N* 
_output_shapes
::
ž
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/ShapeN gradients/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
Ů
gradients/concat_1_grad/SliceSlice,gradients/ResidualRegress/MatMul_grad/MatMul$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/ShapeN*
T0*
Index0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
gradients/concat_1_grad/Slice_1Slice,gradients/ResidualRegress/MatMul_grad/MatMul&gradients/concat_1_grad/ConcatOffset:1 gradients/concat_1_grad/ShapeN:1*
T0*
Index0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
W
gradients_1/ShapeShapeadd_14*
T0*
out_type0*
_output_shapes
:
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_1/add_14_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
o
gradients_1/add_14_grad/Shape_1ShapeTrResidual_2/mul*
T0*
out_type0*
_output_shapes
:
Ă
-gradients_1/add_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_14_grad/Shapegradients_1/add_14_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
gradients_1/add_14_grad/SumSumgradients_1/Fill-gradients_1/add_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ś
gradients_1/add_14_grad/ReshapeReshapegradients_1/add_14_grad/Sumgradients_1/add_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_1/add_14_grad/Sum_1Sumgradients_1/Fill/gradients_1/add_14_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ź
!gradients_1/add_14_grad/Reshape_1Reshapegradients_1/add_14_grad/Sum_1gradients_1/add_14_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
'gradients_1/TrResidual_2/mul_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
y
)gradients_1/TrResidual_2/mul_grad/Shape_1ShapeTrResidual_2/Log*
T0*
out_type0*
_output_shapes
:
á
7gradients_1/TrResidual_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients_1/TrResidual_2/mul_grad/Shape)gradients_1/TrResidual_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

%gradients_1/TrResidual_2/mul_grad/MulMul!gradients_1/add_14_grad/Reshape_1TrResidual_2/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
%gradients_1/TrResidual_2/mul_grad/SumSum%gradients_1/TrResidual_2/mul_grad/Mul7gradients_1/TrResidual_2/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ˇ
)gradients_1/TrResidual_2/mul_grad/ReshapeReshape%gradients_1/TrResidual_2/mul_grad/Sum'gradients_1/TrResidual_2/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

'gradients_1/TrResidual_2/mul_grad/Mul_1MulTrResidual_2/truediv!gradients_1/add_14_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
'gradients_1/TrResidual_2/mul_grad/Sum_1Sum'gradients_1/TrResidual_2/mul_grad/Mul_19gradients_1/TrResidual_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ę
+gradients_1/TrResidual_2/mul_grad/Reshape_1Reshape'gradients_1/TrResidual_2/mul_grad/Sum_1)gradients_1/TrResidual_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
,gradients_1/TrResidual_2/Log_grad/Reciprocal
ReciprocalTrResidual_2/truediv_1,^gradients_1/TrResidual_2/mul_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
%gradients_1/TrResidual_2/Log_grad/mulMul+gradients_1/TrResidual_2/mul_grad/Reshape_1,gradients_1/TrResidual_2/Log_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_1/TrResidual_2/truediv_1_grad/ShapeShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:

/gradients_1/TrResidual_2/truediv_1_grad/Shape_1ShapeTrResidual_2/add*
T0*
out_type0*
_output_shapes
:
ó
=gradients_1/TrResidual_2/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_1/TrResidual_2/truediv_1_grad/Shape/gradients_1/TrResidual_2/truediv_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
/gradients_1/TrResidual_2/truediv_1_grad/RealDivRealDiv%gradients_1/TrResidual_2/Log_grad/mulTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
+gradients_1/TrResidual_2/truediv_1_grad/SumSum/gradients_1/TrResidual_2/truediv_1_grad/RealDiv=gradients_1/TrResidual_2/truediv_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ö
/gradients_1/TrResidual_2/truediv_1_grad/ReshapeReshape+gradients_1/TrResidual_2/truediv_1_grad/Sum-gradients_1/TrResidual_2/truediv_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
+gradients_1/TrResidual_2/truediv_1_grad/NegNegResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
1gradients_1/TrResidual_2/truediv_1_grad/RealDiv_1RealDiv+gradients_1/TrResidual_2/truediv_1_grad/NegTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
1gradients_1/TrResidual_2/truediv_1_grad/RealDiv_2RealDiv1gradients_1/TrResidual_2/truediv_1_grad/RealDiv_1TrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
+gradients_1/TrResidual_2/truediv_1_grad/mulMul%gradients_1/TrResidual_2/Log_grad/mul1gradients_1/TrResidual_2/truediv_1_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
-gradients_1/TrResidual_2/truediv_1_grad/Sum_1Sum+gradients_1/TrResidual_2/truediv_1_grad/mul?gradients_1/TrResidual_2/truediv_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ü
1gradients_1/TrResidual_2/truediv_1_grad/Reshape_1Reshape-gradients_1/TrResidual_2/truediv_1_grad/Sum_1/gradients_1/TrResidual_2/truediv_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
'gradients_1/TrResidual_2/add_grad/ShapeShapeTrResidual_2/sub*
T0*
out_type0*
_output_shapes
:
l
)gradients_1/TrResidual_2/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
á
7gradients_1/TrResidual_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients_1/TrResidual_2/add_grad/Shape)gradients_1/TrResidual_2/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ř
%gradients_1/TrResidual_2/add_grad/SumSum1gradients_1/TrResidual_2/truediv_1_grad/Reshape_17gradients_1/TrResidual_2/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ä
)gradients_1/TrResidual_2/add_grad/ReshapeReshape%gradients_1/TrResidual_2/add_grad/Sum'gradients_1/TrResidual_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
'gradients_1/TrResidual_2/add_grad/Sum_1Sum1gradients_1/TrResidual_2/truediv_1_grad/Reshape_19gradients_1/TrResidual_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
š
+gradients_1/TrResidual_2/add_grad/Reshape_1Reshape'gradients_1/TrResidual_2/add_grad/Sum_1)gradients_1/TrResidual_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
j
'gradients_1/TrResidual_2/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

)gradients_1/TrResidual_2/sub_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
á
7gradients_1/TrResidual_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients_1/TrResidual_2/sub_grad/Shape)gradients_1/TrResidual_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Đ
%gradients_1/TrResidual_2/sub_grad/SumSum)gradients_1/TrResidual_2/add_grad/Reshape7gradients_1/TrResidual_2/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ł
)gradients_1/TrResidual_2/sub_grad/ReshapeReshape%gradients_1/TrResidual_2/sub_grad/Sum'gradients_1/TrResidual_2/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ô
'gradients_1/TrResidual_2/sub_grad/Sum_1Sum)gradients_1/TrResidual_2/add_grad/Reshape9gradients_1/TrResidual_2/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
x
%gradients_1/TrResidual_2/sub_grad/NegNeg'gradients_1/TrResidual_2/sub_grad/Sum_1*
T0*
_output_shapes
:
Č
+gradients_1/TrResidual_2/sub_grad/Reshape_1Reshape%gradients_1/TrResidual_2/sub_grad/Neg)gradients_1/TrResidual_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
gradients_1/AddNAddN/gradients_1/TrResidual_2/truediv_1_grad/Reshape+gradients_1/TrResidual_2/sub_grad/Reshape_1*
T0*B
_class8
64loc:@gradients_1/TrResidual_2/truediv_1_grad/Reshape*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
4gradients_1/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoidgradients_1/AddN*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_1/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
x
.gradients_1/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_1/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/ResidualRegress/add_3_grad/Shape.gradients_1/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
*gradients_1/ResidualRegress/add_3_grad/SumSum4gradients_1/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients_1/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ó
.gradients_1/ResidualRegress/add_3_grad/ReshapeReshape*gradients_1/ResidualRegress/add_3_grad/Sum,gradients_1/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
,gradients_1/ResidualRegress/add_3_grad/Sum_1Sum4gradients_1/ResidualRegress/Sigmoid_grad/SigmoidGrad>gradients_1/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_1/ResidualRegress/add_3_grad/Reshape_1Reshape,gradients_1/ResidualRegress/add_3_grad/Sum_1.gradients_1/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
ŕ
0gradients_1/ResidualRegress/MatMul_3_grad/MatMulMatMul.gradients_1/ResidualRegress/add_3_grad/ReshapeResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
2gradients_1/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2.gradients_1/ResidualRegress/add_3_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
š
0gradients_1/ResidualRegress/Relu_2_grad/ReluGradReluGrad0gradients_1/ResidualRegress/MatMul_3_grad/MatMulResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_1/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
y
.gradients_1/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_1/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/ResidualRegress/add_2_grad/Shape.gradients_1/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_1/ResidualRegress/add_2_grad/SumSum0gradients_1/ResidualRegress/Relu_2_grad/ReluGrad<gradients_1/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_1/ResidualRegress/add_2_grad/ReshapeReshape*gradients_1/ResidualRegress/add_2_grad/Sum,gradients_1/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_1/ResidualRegress/add_2_grad/Sum_1Sum0gradients_1/ResidualRegress/Relu_2_grad/ReluGrad>gradients_1/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_1/ResidualRegress/add_2_grad/Reshape_1Reshape,gradients_1/ResidualRegress/add_2_grad/Sum_1.gradients_1/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
ŕ
0gradients_1/ResidualRegress/MatMul_2_grad/MatMulMatMul.gradients_1/ResidualRegress/add_2_grad/ReshapeResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
2gradients_1/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1.gradients_1/ResidualRegress/add_2_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

š
0gradients_1/ResidualRegress/Relu_1_grad/ReluGradReluGrad0gradients_1/ResidualRegress/MatMul_2_grad/MatMulResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_1/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
y
.gradients_1/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_1/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/ResidualRegress/add_1_grad/Shape.gradients_1/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_1/ResidualRegress/add_1_grad/SumSum0gradients_1/ResidualRegress/Relu_1_grad/ReluGrad<gradients_1/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_1/ResidualRegress/add_1_grad/ReshapeReshape*gradients_1/ResidualRegress/add_1_grad/Sum,gradients_1/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_1/ResidualRegress/add_1_grad/Sum_1Sum0gradients_1/ResidualRegress/Relu_1_grad/ReluGrad>gradients_1/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_1/ResidualRegress/add_1_grad/Reshape_1Reshape,gradients_1/ResidualRegress/add_1_grad/Sum_1.gradients_1/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
ŕ
0gradients_1/ResidualRegress/MatMul_1_grad/MatMulMatMul.gradients_1/ResidualRegress/add_1_grad/ReshapeResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
2gradients_1/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu.gradients_1/ResidualRegress/add_1_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

ľ
.gradients_1/ResidualRegress/Relu_grad/ReluGradReluGrad0gradients_1/ResidualRegress/MatMul_1_grad/MatMulResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients_1/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients_1/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients_1/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/ResidualRegress/add_grad/Shape,gradients_1/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients_1/ResidualRegress/add_grad/SumSum.gradients_1/ResidualRegress/Relu_grad/ReluGrad:gradients_1/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients_1/ResidualRegress/add_grad/ReshapeReshape(gradients_1/ResidualRegress/add_grad/Sum*gradients_1/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients_1/ResidualRegress/add_grad/Sum_1Sum.gradients_1/ResidualRegress/Relu_grad/ReluGrad<gradients_1/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients_1/ResidualRegress/add_grad/Reshape_1Reshape*gradients_1/ResidualRegress/add_grad/Sum_1,gradients_1/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ű
.gradients_1/ResidualRegress/MatMul_grad/MatMulMatMul,gradients_1/ResidualRegress/add_grad/ReshapeResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Â
0gradients_1/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1,gradients_1/ResidualRegress/add_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
`
gradients_1/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
y
gradients_1/concat_1_grad/modFloorModconcat_1/axisgradients_1/concat_1_grad/Rank*
T0*
_output_shapes
: 
e
gradients_1/concat_1_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:

 gradients_1/concat_1_grad/ShapeNShapeNconcatPlaceholder*
T0*
out_type0*
N* 
_output_shapes
::
Ć
&gradients_1/concat_1_grad/ConcatOffsetConcatOffsetgradients_1/concat_1_grad/mod gradients_1/concat_1_grad/ShapeN"gradients_1/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
á
gradients_1/concat_1_grad/SliceSlice.gradients_1/ResidualRegress/MatMul_grad/MatMul&gradients_1/concat_1_grad/ConcatOffset gradients_1/concat_1_grad/ShapeN*
T0*
Index0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
!gradients_1/concat_1_grad/Slice_1Slice.gradients_1/ResidualRegress/MatMul_grad/MatMul(gradients_1/concat_1_grad/ConcatOffset:1"gradients_1/concat_1_grad/ShapeN:1*
T0*
Index0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
{
Abs_8/xPackgradients/concat_1_grad/Slice_1*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
K
Abs_8AbsAbs_8/x*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
]
Const_14Const*!
valueB"          *
dtype0*
_output_shapes
:
Y
MaxMaxAbs_8Const_14*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
}
Abs_9/xPack!gradients_1/concat_1_grad/Slice_1*
T0*

axis *
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
K
Abs_9AbsAbs_9/x*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
]
Const_15Const*!
valueB"          *
dtype0*
_output_shapes
:
[
Max_1MaxAbs_9Const_15*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:

moments/meanMeangradients/concat_1_grad/Slice_1moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:2
[
moments/StopGradientStopGradientmoments/mean*
T0*
_output_shapes

:2

moments/SquaredDifferenceSquaredDifferencegradients/concat_1_grad/Slice_1moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
l
"moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:

moments/varianceMeanmoments/SquaredDifference"moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:2
d
moments/SqueezeSqueezemoments/mean*
squeeze_dims
 *
T0*
_output_shapes
:2
j
moments/Squeeze_1Squeezemoments/variance*
squeeze_dims
 *
T0*
_output_shapes
:2
j
 moments_1/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ą
moments_1/meanMean!gradients_1/concat_1_grad/Slice_1 moments_1/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:2
_
moments_1/StopGradientStopGradientmoments_1/mean*
T0*
_output_shapes

:2

moments_1/SquaredDifferenceSquaredDifference!gradients_1/concat_1_grad/Slice_1moments_1/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
n
$moments_1/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ł
moments_1/varianceMeanmoments_1/SquaredDifference$moments_1/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:2
h
moments_1/SqueezeSqueezemoments_1/mean*
squeeze_dims
 *
T0*
_output_shapes
:2
n
moments_1/Squeeze_1Squeezemoments_1/variance*
squeeze_dims
 *
T0*
_output_shapes
:2
U
sub_8SubPlaceholder_3add_15*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
Square_6Squaresub_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Const_16Const*
valueB"       *
dtype0*
_output_shapes
:
`
Mean_6MeanSquare_6Const_16*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
=
add_32AddMean_6Max_1*
T0*
_output_shapes
: 
T
gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_2/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
C
(gradients_2/add_17_grad/tuple/group_depsNoOp^gradients_2/Fill
ż
0gradients_2/add_17_grad/tuple/control_dependencyIdentitygradients_2/Fill)^gradients_2/add_17_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
Á
2gradients_2/add_17_grad/tuple/control_dependency_1Identitygradients_2/Fill)^gradients_2/add_17_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
c
(gradients_2/add_16_grad/tuple/group_depsNoOp1^gradients_2/add_17_grad/tuple/control_dependency
ß
0gradients_2/add_16_grad/tuple/control_dependencyIdentity0gradients_2/add_17_grad/tuple/control_dependency)^gradients_2/add_16_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
á
2gradients_2/add_16_grad/tuple/control_dependency_1Identity0gradients_2/add_17_grad/tuple/control_dependency)^gradients_2/add_16_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
|
9gradients_2/softmax_cross_entropy_loss_2/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
~
;gradients_2/softmax_cross_entropy_loss_2/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Igradients_2/softmax_cross_entropy_loss_2/value_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_2/softmax_cross_entropy_loss_2/value_grad/Shape;gradients_2/softmax_cross_entropy_loss_2/value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
>gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nanDivNoNan2gradients_2/add_17_grad/tuple/control_dependency_1(softmax_cross_entropy_loss_2/num_present*
T0*
_output_shapes
: 

7gradients_2/softmax_cross_entropy_loss_2/value_grad/SumSum>gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nanIgradients_2/softmax_cross_entropy_loss_2/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
é
;gradients_2/softmax_cross_entropy_loss_2/value_grad/ReshapeReshape7gradients_2/softmax_cross_entropy_loss_2/value_grad/Sum9gradients_2/softmax_cross_entropy_loss_2/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

7gradients_2/softmax_cross_entropy_loss_2/value_grad/NegNeg"softmax_cross_entropy_loss_2/Sum_1*
T0*
_output_shapes
: 
Đ
@gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nan_1DivNoNan7gradients_2/softmax_cross_entropy_loss_2/value_grad/Neg(softmax_cross_entropy_loss_2/num_present*
T0*
_output_shapes
: 
Ů
@gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nan_2DivNoNan@gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nan_1(softmax_cross_entropy_loss_2/num_present*
T0*
_output_shapes
: 
Ő
7gradients_2/softmax_cross_entropy_loss_2/value_grad/mulMul2gradients_2/add_17_grad/tuple/control_dependency_1@gradients_2/softmax_cross_entropy_loss_2/value_grad/div_no_nan_2*
T0*
_output_shapes
: 

9gradients_2/softmax_cross_entropy_loss_2/value_grad/Sum_1Sum7gradients_2/softmax_cross_entropy_loss_2/value_grad/mulKgradients_2/softmax_cross_entropy_loss_2/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ď
=gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape_1Reshape9gradients_2/softmax_cross_entropy_loss_2/value_grad/Sum_1;gradients_2/softmax_cross_entropy_loss_2/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ę
Dgradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/group_depsNoOp<^gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape>^gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape_1
Í
Lgradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/control_dependencyIdentity;gradients_2/softmax_cross_entropy_loss_2/value_grad/ReshapeE^gradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape*
_output_shapes
: 
Ó
Ngradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/control_dependency_1Identity=gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape_1E^gradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_2/softmax_cross_entropy_loss_2/value_grad/Reshape_1*
_output_shapes
: 
z
7gradients_2/softmax_cross_entropy_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
|
9gradients_2/softmax_cross_entropy_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Ggradients_2/softmax_cross_entropy_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_2/softmax_cross_entropy_loss/value_grad/Shape9gradients_2/softmax_cross_entropy_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ă
<gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nanDivNoNan0gradients_2/add_16_grad/tuple/control_dependency&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 

5gradients_2/softmax_cross_entropy_loss/value_grad/SumSum<gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nanGgradients_2/softmax_cross_entropy_loss/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ă
9gradients_2/softmax_cross_entropy_loss/value_grad/ReshapeReshape5gradients_2/softmax_cross_entropy_loss/value_grad/Sum7gradients_2/softmax_cross_entropy_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

5gradients_2/softmax_cross_entropy_loss/value_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
Ę
>gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nan_1DivNoNan5gradients_2/softmax_cross_entropy_loss/value_grad/Neg&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
Ó
>gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nan_2DivNoNan>gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nan_1&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
Ď
5gradients_2/softmax_cross_entropy_loss/value_grad/mulMul0gradients_2/add_16_grad/tuple/control_dependency>gradients_2/softmax_cross_entropy_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
ţ
7gradients_2/softmax_cross_entropy_loss/value_grad/Sum_1Sum5gradients_2/softmax_cross_entropy_loss/value_grad/mulIgradients_2/softmax_cross_entropy_loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
é
;gradients_2/softmax_cross_entropy_loss/value_grad/Reshape_1Reshape7gradients_2/softmax_cross_entropy_loss/value_grad/Sum_19gradients_2/softmax_cross_entropy_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ä
Bgradients_2/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp:^gradients_2/softmax_cross_entropy_loss/value_grad/Reshape<^gradients_2/softmax_cross_entropy_loss/value_grad/Reshape_1
Ĺ
Jgradients_2/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity9gradients_2/softmax_cross_entropy_loss/value_grad/ReshapeC^gradients_2/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/softmax_cross_entropy_loss/value_grad/Reshape*
_output_shapes
: 
Ë
Lgradients_2/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity;gradients_2/softmax_cross_entropy_loss/value_grad/Reshape_1C^gradients_2/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/softmax_cross_entropy_loss/value_grad/Reshape_1*
_output_shapes
: 
|
9gradients_2/softmax_cross_entropy_loss_1/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
~
;gradients_2/softmax_cross_entropy_loss_1/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Igradients_2/softmax_cross_entropy_loss_1/value_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_2/softmax_cross_entropy_loss_1/value_grad/Shape;gradients_2/softmax_cross_entropy_loss_1/value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
>gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nanDivNoNan2gradients_2/add_16_grad/tuple/control_dependency_1(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 

7gradients_2/softmax_cross_entropy_loss_1/value_grad/SumSum>gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nanIgradients_2/softmax_cross_entropy_loss_1/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
é
;gradients_2/softmax_cross_entropy_loss_1/value_grad/ReshapeReshape7gradients_2/softmax_cross_entropy_loss_1/value_grad/Sum9gradients_2/softmax_cross_entropy_loss_1/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

7gradients_2/softmax_cross_entropy_loss_1/value_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
Đ
@gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nan_1DivNoNan7gradients_2/softmax_cross_entropy_loss_1/value_grad/Neg(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
Ů
@gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nan_2DivNoNan@gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nan_1(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
Ő
7gradients_2/softmax_cross_entropy_loss_1/value_grad/mulMul2gradients_2/add_16_grad/tuple/control_dependency_1@gradients_2/softmax_cross_entropy_loss_1/value_grad/div_no_nan_2*
T0*
_output_shapes
: 

9gradients_2/softmax_cross_entropy_loss_1/value_grad/Sum_1Sum7gradients_2/softmax_cross_entropy_loss_1/value_grad/mulKgradients_2/softmax_cross_entropy_loss_1/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ď
=gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape_1Reshape9gradients_2/softmax_cross_entropy_loss_1/value_grad/Sum_1;gradients_2/softmax_cross_entropy_loss_1/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ę
Dgradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp<^gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape>^gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape_1
Í
Lgradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity;gradients_2/softmax_cross_entropy_loss_1/value_grad/ReshapeE^gradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape*
_output_shapes
: 
Ó
Ngradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity=gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape_1E^gradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_2/softmax_cross_entropy_loss_1/value_grad/Reshape_1*
_output_shapes
: 

Agradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

;gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/ReshapeReshapeLgradients_2/softmax_cross_entropy_loss_2/value_grad/tuple/control_dependencyAgradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
9gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ë
8gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/TileTile;gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/Reshape9gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

?gradients_2/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

9gradients_2/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeJgradients_2/softmax_cross_entropy_loss/value_grad/tuple/control_dependency?gradients_2/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
z
7gradients_2/softmax_cross_entropy_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ĺ
6gradients_2/softmax_cross_entropy_loss/Sum_1_grad/TileTile9gradients_2/softmax_cross_entropy_loss/Sum_1_grad/Reshape7gradients_2/softmax_cross_entropy_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

Agradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

;gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeLgradients_2/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyAgradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
|
9gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
ë
8gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile;gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape9gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

?gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ň
9gradients_2/softmax_cross_entropy_loss_2/Sum_grad/ReshapeReshape8gradients_2/softmax_cross_entropy_loss_2/Sum_1_grad/Tile?gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

7gradients_2/softmax_cross_entropy_loss_2/Sum_grad/ShapeShape softmax_cross_entropy_loss_2/Mul*
T0*
out_type0*
_output_shapes
:
ň
6gradients_2/softmax_cross_entropy_loss_2/Sum_grad/TileTile9gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Reshape7gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ě
7gradients_2/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape6gradients_2/softmax_cross_entropy_loss/Sum_1_grad/Tile=gradients_2/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

5gradients_2/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
T0*
out_type0*
_output_shapes
:
ě
4gradients_2/softmax_cross_entropy_loss/Sum_grad/TileTile7gradients_2/softmax_cross_entropy_loss/Sum_grad/Reshape5gradients_2/softmax_cross_entropy_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ň
9gradients_2/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape8gradients_2/softmax_cross_entropy_loss_1/Sum_1_grad/Tile?gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

7gradients_2/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
T0*
out_type0*
_output_shapes
:
ň
6gradients_2/softmax_cross_entropy_loss_1/Sum_grad/TileTile9gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Reshape7gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/ShapeShape/softmax_cross_entropy_loss_2/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:
|
9gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Ggradients_2/softmax_cross_entropy_loss_2/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Shape9gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
5gradients_2/softmax_cross_entropy_loss_2/Mul_grad/MulMul6gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Tile&softmax_cross_entropy_loss_2/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
5gradients_2/softmax_cross_entropy_loss_2/Mul_grad/SumSum5gradients_2/softmax_cross_entropy_loss_2/Mul_grad/MulGgradients_2/softmax_cross_entropy_loss_2/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
đ
9gradients_2/softmax_cross_entropy_loss_2/Mul_grad/ReshapeReshape5gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Sum7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Mul_1Mul/softmax_cross_entropy_loss_2/xentropy/Reshape_26gradients_2/softmax_cross_entropy_loss_2/Sum_grad/Tile*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Sum_1Sum7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Mul_1Igradients_2/softmax_cross_entropy_loss_2/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
é
;gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape_1Reshape7gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Sum_19gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ä
Bgradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/group_depsNoOp:^gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape<^gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape_1
Ň
Jgradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/control_dependencyIdentity9gradients_2/softmax_cross_entropy_loss_2/Mul_grad/ReshapeC^gradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
Lgradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/control_dependency_1Identity;gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape_1C^gradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/softmax_cross_entropy_loss_2/Mul_grad/Reshape_1*
_output_shapes
: 
˘
5gradients_2/softmax_cross_entropy_loss/Mul_grad/ShapeShape-softmax_cross_entropy_loss/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:
z
7gradients_2/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Egradients_2/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_2/softmax_cross_entropy_loss/Mul_grad/Shape7gradients_2/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ä
3gradients_2/softmax_cross_entropy_loss/Mul_grad/MulMul4gradients_2/softmax_cross_entropy_loss/Sum_grad/Tile$softmax_cross_entropy_loss/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
3gradients_2/softmax_cross_entropy_loss/Mul_grad/SumSum3gradients_2/softmax_cross_entropy_loss/Mul_grad/MulEgradients_2/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ę
7gradients_2/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape3gradients_2/softmax_cross_entropy_loss/Mul_grad/Sum5gradients_2/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
5gradients_2/softmax_cross_entropy_loss/Mul_grad/Mul_1Mul-softmax_cross_entropy_loss/xentropy/Reshape_24gradients_2/softmax_cross_entropy_loss/Sum_grad/Tile*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
5gradients_2/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum5gradients_2/softmax_cross_entropy_loss/Mul_grad/Mul_1Ggradients_2/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ă
9gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape5gradients_2/softmax_cross_entropy_loss/Mul_grad/Sum_17gradients_2/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ž
@gradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp8^gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape:^gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape_1
Ę
Hgradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity7gradients_2/softmax_cross_entropy_loss/Mul_grad/ReshapeA^gradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Jgradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity9gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape_1A^gradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: 
Ś
7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape/softmax_cross_entropy_loss_1/xentropy/Reshape_2*
T0*
out_type0*
_output_shapes
:
|
9gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Ggradients_2/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Shape9gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
5gradients_2/softmax_cross_entropy_loss_1/Mul_grad/MulMul6gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Tile&softmax_cross_entropy_loss_1/ToFloat/x*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
5gradients_2/softmax_cross_entropy_loss_1/Mul_grad/SumSum5gradients_2/softmax_cross_entropy_loss_1/Mul_grad/MulGgradients_2/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
đ
9gradients_2/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape5gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Sum7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Mul_1Mul/softmax_cross_entropy_loss_1/xentropy/Reshape_26gradients_2/softmax_cross_entropy_loss_1/Sum_grad/Tile*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Mul_1Igradients_2/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
é
;gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape7gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Sum_19gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ä
Bgradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp:^gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape<^gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
Ň
Jgradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity9gradients_2/softmax_cross_entropy_loss_1/Mul_grad/ReshapeC^gradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
Lgradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity;gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1C^gradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
Ť
Fgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_2/xentropy*
T0*
out_type0*
_output_shapes
:
Ł
Hgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_2_grad/ReshapeReshapeJgradients_2/softmax_cross_entropy_loss_2/Mul_grad/tuple/control_dependencyFgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Dgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
T0*
out_type0*
_output_shapes
:

Fgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/ReshapeReshapeHgradients_2/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyDgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Fgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
T0*
out_type0*
_output_shapes
:
Ł
Hgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_2_grad/ReshapeReshapeJgradients_2/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyFgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_2/zeros_like	ZerosLike'softmax_cross_entropy_loss_2/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Egradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

Agradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims
ExpandDimsHgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_2_grad/ReshapeEgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
:gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mulMulAgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims'softmax_cross_entropy_loss_2/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
š
Agradients_2/softmax_cross_entropy_loss_2/xentropy_grad/LogSoftmax
LogSoftmax-softmax_cross_entropy_loss_2/xentropy/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ż
:gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/NegNegAgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Ggradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
˘
Cgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims_1
ExpandDimsHgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_2_grad/ReshapeGgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
<gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul_1MulCgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/ExpandDims_1:gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ë
Ggradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/group_depsNoOp;^gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul=^gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul_1
ë
Ogradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/control_dependencyIdentity:gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mulH^gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ń
Qgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/control_dependency_1Identity<gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul_1H^gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_2/softmax_cross_entropy_loss_2/xentropy_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients_2/zeros_like_1	ZerosLike%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Cgradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

?gradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDimsFgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/ReshapeCgradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
8gradients_2/softmax_cross_entropy_loss/xentropy_grad/mulMul?gradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ľ
?gradients_2/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax
LogSoftmax+softmax_cross_entropy_loss/xentropy/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ť
8gradients_2/softmax_cross_entropy_loss/xentropy_grad/NegNeg?gradients_2/softmax_cross_entropy_loss/xentropy_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Egradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

Agradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1
ExpandDimsFgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_2_grad/ReshapeEgradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
:gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul_1MulAgradients_2/softmax_cross_entropy_loss/xentropy_grad/ExpandDims_18gradients_2/softmax_cross_entropy_loss/xentropy_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ĺ
Egradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/group_depsNoOp9^gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul;^gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul_1
ă
Mgradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/control_dependencyIdentity8gradients_2/softmax_cross_entropy_loss/xentropy_grad/mulF^gradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
é
Ogradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/control_dependency_1Identity:gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul_1F^gradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/softmax_cross_entropy_loss/xentropy_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients_2/zeros_like_2	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Egradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

Agradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDimsHgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_2_grad/ReshapeEgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
:gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mulMulAgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
š
Agradients_2/softmax_cross_entropy_loss_1/xentropy_grad/LogSoftmax
LogSoftmax-softmax_cross_entropy_loss_1/xentropy/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ż
:gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/NegNegAgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Ggradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
˘
Cgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims_1
ExpandDimsHgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_2_grad/ReshapeGgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
<gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul_1MulCgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims_1:gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ë
Ggradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/group_depsNoOp;^gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul=^gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul_1
ë
Ogradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/control_dependencyIdentity:gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mulH^gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ń
Qgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/control_dependency_1Identity<gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul_1H^gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_2/softmax_cross_entropy_loss_1/xentropy_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Dgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_grad/ShapeShapestrided_slice_11*
T0*
out_type0*
_output_shapes
:
¨
Fgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_grad/ReshapeReshapeOgradients_2/softmax_cross_entropy_loss_2/xentropy_grad/tuple/control_dependencyDgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Bgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_grad/ShapeShapestrided_slice_7*
T0*
out_type0*
_output_shapes
:
˘
Dgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_grad/ReshapeReshapeMgradients_2/softmax_cross_entropy_loss/xentropy_grad/tuple/control_dependencyBgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_grad/ShapeShapestrided_slice_9*
T0*
out_type0*
_output_shapes
:
¨
Fgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_grad/ReshapeReshapeOgradients_2/softmax_cross_entropy_loss_1/xentropy_grad/tuple/control_dependencyDgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
'gradients_2/strided_slice_11_grad/ShapeShapeClassify/concat*
T0*
out_type0*
_output_shapes
:
Ş
2gradients_2/strided_slice_11_grad/StridedSliceGradStridedSliceGrad'gradients_2/strided_slice_11_grad/Shapestrided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2Fgradients_2/softmax_cross_entropy_loss_2/xentropy/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
&gradients_2/strided_slice_7_grad/ShapeShapeClassify/concat*
T0*
out_type0*
_output_shapes
:
Ł
1gradients_2/strided_slice_7_grad/StridedSliceGradStridedSliceGrad&gradients_2/strided_slice_7_grad/Shapestrided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2Dgradients_2/softmax_cross_entropy_loss/xentropy/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
&gradients_2/strided_slice_9_grad/ShapeShapeClassify/concat*
T0*
out_type0*
_output_shapes
:
Ľ
1gradients_2/strided_slice_9_grad/StridedSliceGradStridedSliceGrad&gradients_2/strided_slice_9_grad/Shapestrided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2Fgradients_2/softmax_cross_entropy_loss_1/xentropy/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
gradients_2/AddNAddN2gradients_2/strided_slice_11_grad/StridedSliceGrad1gradients_2/strided_slice_7_grad/StridedSliceGrad1gradients_2/strided_slice_9_grad/StridedSliceGrad*
T0*E
_class;
97loc:@gradients_2/strided_slice_11_grad/StridedSliceGrad*
N*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
%gradients_2/Classify/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 

$gradients_2/Classify/concat_grad/modFloorModClassify/concat/axis%gradients_2/Classify/concat_grad/Rank*
T0*
_output_shapes
: 
y
&gradients_2/Classify/concat_grad/ShapeShapeClassify/ExpandDims*
T0*
out_type0*
_output_shapes
:
ž
'gradients_2/Classify/concat_grad/ShapeNShapeNClassify/ExpandDimsClassify/ExpandDims_1Classify/ExpandDims_2*
T0*
out_type0*
N*&
_output_shapes
:::

-gradients_2/Classify/concat_grad/ConcatOffsetConcatOffset$gradients_2/Classify/concat_grad/mod'gradients_2/Classify/concat_grad/ShapeN)gradients_2/Classify/concat_grad/ShapeN:1)gradients_2/Classify/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
Ü
&gradients_2/Classify/concat_grad/SliceSlicegradients_2/AddN-gradients_2/Classify/concat_grad/ConcatOffset'gradients_2/Classify/concat_grad/ShapeN*
T0*
Index0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
(gradients_2/Classify/concat_grad/Slice_1Slicegradients_2/AddN/gradients_2/Classify/concat_grad/ConcatOffset:1)gradients_2/Classify/concat_grad/ShapeN:1*
T0*
Index0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
(gradients_2/Classify/concat_grad/Slice_2Slicegradients_2/AddN/gradients_2/Classify/concat_grad/ConcatOffset:2)gradients_2/Classify/concat_grad/ShapeN:2*
T0*
Index0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
1gradients_2/Classify/concat_grad/tuple/group_depsNoOp'^gradients_2/Classify/concat_grad/Slice)^gradients_2/Classify/concat_grad/Slice_1)^gradients_2/Classify/concat_grad/Slice_2

9gradients_2/Classify/concat_grad/tuple/control_dependencyIdentity&gradients_2/Classify/concat_grad/Slice2^gradients_2/Classify/concat_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_2/Classify/concat_grad/Slice*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_2/Classify/concat_grad/tuple/control_dependency_1Identity(gradients_2/Classify/concat_grad/Slice_12^gradients_2/Classify/concat_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_2/Classify/concat_grad/Slice_1*+
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_2/Classify/concat_grad/tuple/control_dependency_2Identity(gradients_2/Classify/concat_grad/Slice_22^gradients_2/Classify/concat_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_2/Classify/concat_grad/Slice_2*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
*gradients_2/Classify/ExpandDims_grad/ShapeShapeClassify/add_3*
T0*
out_type0*
_output_shapes
:
Ţ
,gradients_2/Classify/ExpandDims_grad/ReshapeReshape9gradients_2/Classify/concat_grad/tuple/control_dependency*gradients_2/Classify/ExpandDims_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
,gradients_2/Classify/ExpandDims_1_grad/ShapeShapeClassify/add_4*
T0*
out_type0*
_output_shapes
:
ä
.gradients_2/Classify/ExpandDims_1_grad/ReshapeReshape;gradients_2/Classify/concat_grad/tuple/control_dependency_1,gradients_2/Classify/ExpandDims_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
,gradients_2/Classify/ExpandDims_2_grad/ShapeShapeClassify/add_5*
T0*
out_type0*
_output_shapes
:
ä
.gradients_2/Classify/ExpandDims_2_grad/ReshapeReshape;gradients_2/Classify/concat_grad/tuple/control_dependency_2,gradients_2/Classify/ExpandDims_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
%gradients_2/Classify/add_3_grad/ShapeShapeClassify/MatMul_3*
T0*
out_type0*
_output_shapes
:
q
'gradients_2/Classify/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ű
5gradients_2/Classify/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_2/Classify/add_3_grad/Shape'gradients_2/Classify/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
#gradients_2/Classify/add_3_grad/SumSum,gradients_2/Classify/ExpandDims_grad/Reshape5gradients_2/Classify/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
'gradients_2/Classify/add_3_grad/ReshapeReshape#gradients_2/Classify/add_3_grad/Sum%gradients_2/Classify/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
%gradients_2/Classify/add_3_grad/Sum_1Sum,gradients_2/Classify/ExpandDims_grad/Reshape7gradients_2/Classify/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ˇ
)gradients_2/Classify/add_3_grad/Reshape_1Reshape%gradients_2/Classify/add_3_grad/Sum_1'gradients_2/Classify/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

0gradients_2/Classify/add_3_grad/tuple/group_depsNoOp(^gradients_2/Classify/add_3_grad/Reshape*^gradients_2/Classify/add_3_grad/Reshape_1

8gradients_2/Classify/add_3_grad/tuple/control_dependencyIdentity'gradients_2/Classify/add_3_grad/Reshape1^gradients_2/Classify/add_3_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_2/Classify/add_3_grad/tuple/control_dependency_1Identity)gradients_2/Classify/add_3_grad/Reshape_11^gradients_2/Classify/add_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/add_3_grad/Reshape_1*
_output_shapes
:
v
%gradients_2/Classify/add_4_grad/ShapeShapeClassify/MatMul_4*
T0*
out_type0*
_output_shapes
:
q
'gradients_2/Classify/add_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ű
5gradients_2/Classify/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_2/Classify/add_4_grad/Shape'gradients_2/Classify/add_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ń
#gradients_2/Classify/add_4_grad/SumSum.gradients_2/Classify/ExpandDims_1_grad/Reshape5gradients_2/Classify/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
'gradients_2/Classify/add_4_grad/ReshapeReshape#gradients_2/Classify/add_4_grad/Sum%gradients_2/Classify/add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
%gradients_2/Classify/add_4_grad/Sum_1Sum.gradients_2/Classify/ExpandDims_1_grad/Reshape7gradients_2/Classify/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ˇ
)gradients_2/Classify/add_4_grad/Reshape_1Reshape%gradients_2/Classify/add_4_grad/Sum_1'gradients_2/Classify/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

0gradients_2/Classify/add_4_grad/tuple/group_depsNoOp(^gradients_2/Classify/add_4_grad/Reshape*^gradients_2/Classify/add_4_grad/Reshape_1

8gradients_2/Classify/add_4_grad/tuple/control_dependencyIdentity'gradients_2/Classify/add_4_grad/Reshape1^gradients_2/Classify/add_4_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_4_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_2/Classify/add_4_grad/tuple/control_dependency_1Identity)gradients_2/Classify/add_4_grad/Reshape_11^gradients_2/Classify/add_4_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/add_4_grad/Reshape_1*
_output_shapes
:
v
%gradients_2/Classify/add_5_grad/ShapeShapeClassify/MatMul_5*
T0*
out_type0*
_output_shapes
:
q
'gradients_2/Classify/add_5_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ű
5gradients_2/Classify/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_2/Classify/add_5_grad/Shape'gradients_2/Classify/add_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ń
#gradients_2/Classify/add_5_grad/SumSum.gradients_2/Classify/ExpandDims_2_grad/Reshape5gradients_2/Classify/add_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
'gradients_2/Classify/add_5_grad/ReshapeReshape#gradients_2/Classify/add_5_grad/Sum%gradients_2/Classify/add_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
%gradients_2/Classify/add_5_grad/Sum_1Sum.gradients_2/Classify/ExpandDims_2_grad/Reshape7gradients_2/Classify/add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ˇ
)gradients_2/Classify/add_5_grad/Reshape_1Reshape%gradients_2/Classify/add_5_grad/Sum_1'gradients_2/Classify/add_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

0gradients_2/Classify/add_5_grad/tuple/group_depsNoOp(^gradients_2/Classify/add_5_grad/Reshape*^gradients_2/Classify/add_5_grad/Reshape_1

8gradients_2/Classify/add_5_grad/tuple/control_dependencyIdentity'gradients_2/Classify/add_5_grad/Reshape1^gradients_2/Classify/add_5_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_5_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_2/Classify/add_5_grad/tuple/control_dependency_1Identity)gradients_2/Classify/add_5_grad/Reshape_11^gradients_2/Classify/add_5_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/add_5_grad/Reshape_1*
_output_shapes
:
Ú
)gradients_2/Classify/MatMul_3_grad/MatMulMatMul8gradients_2/Classify/add_3_grad/tuple/control_dependencyClassify/w4_1/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
+gradients_2/Classify/MatMul_3_grad/MatMul_1MatMulClassify/Relu_28gradients_2/Classify/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

3gradients_2/Classify/MatMul_3_grad/tuple/group_depsNoOp*^gradients_2/Classify/MatMul_3_grad/MatMul,^gradients_2/Classify/MatMul_3_grad/MatMul_1

;gradients_2/Classify/MatMul_3_grad/tuple/control_dependencyIdentity)gradients_2/Classify/MatMul_3_grad/MatMul4^gradients_2/Classify/MatMul_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/Classify/MatMul_3_grad/tuple/control_dependency_1Identity+gradients_2/Classify/MatMul_3_grad/MatMul_14^gradients_2/Classify/MatMul_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/Classify/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Ú
)gradients_2/Classify/MatMul_4_grad/MatMulMatMul8gradients_2/Classify/add_4_grad/tuple/control_dependencyClassify/w4_2/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
+gradients_2/Classify/MatMul_4_grad/MatMul_1MatMulClassify/Relu_28gradients_2/Classify/add_4_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

3gradients_2/Classify/MatMul_4_grad/tuple/group_depsNoOp*^gradients_2/Classify/MatMul_4_grad/MatMul,^gradients_2/Classify/MatMul_4_grad/MatMul_1

;gradients_2/Classify/MatMul_4_grad/tuple/control_dependencyIdentity)gradients_2/Classify/MatMul_4_grad/MatMul4^gradients_2/Classify/MatMul_4_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_4_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/Classify/MatMul_4_grad/tuple/control_dependency_1Identity+gradients_2/Classify/MatMul_4_grad/MatMul_14^gradients_2/Classify/MatMul_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/Classify/MatMul_4_grad/MatMul_1*
_output_shapes
:	
Ú
)gradients_2/Classify/MatMul_5_grad/MatMulMatMul8gradients_2/Classify/add_5_grad/tuple/control_dependencyClassify/w4_3/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
+gradients_2/Classify/MatMul_5_grad/MatMul_1MatMulClassify/Relu_28gradients_2/Classify/add_5_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

3gradients_2/Classify/MatMul_5_grad/tuple/group_depsNoOp*^gradients_2/Classify/MatMul_5_grad/MatMul,^gradients_2/Classify/MatMul_5_grad/MatMul_1

;gradients_2/Classify/MatMul_5_grad/tuple/control_dependencyIdentity)gradients_2/Classify/MatMul_5_grad/MatMul4^gradients_2/Classify/MatMul_5_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_5_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/Classify/MatMul_5_grad/tuple/control_dependency_1Identity+gradients_2/Classify/MatMul_5_grad/MatMul_14^gradients_2/Classify/MatMul_5_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/Classify/MatMul_5_grad/MatMul_1*
_output_shapes
:	
Ë
gradients_2/AddN_1AddN;gradients_2/Classify/MatMul_3_grad/tuple/control_dependency;gradients_2/Classify/MatMul_4_grad/tuple/control_dependency;gradients_2/Classify/MatMul_5_grad/tuple/control_dependency*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_3_grad/MatMul*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

)gradients_2/Classify/Relu_2_grad/ReluGradReluGradgradients_2/AddN_1Classify/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
%gradients_2/Classify/add_2_grad/ShapeShapeClassify/MatMul_2*
T0*
out_type0*
_output_shapes
:
r
'gradients_2/Classify/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ű
5gradients_2/Classify/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_2/Classify/add_2_grad/Shape'gradients_2/Classify/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ě
#gradients_2/Classify/add_2_grad/SumSum)gradients_2/Classify/Relu_2_grad/ReluGrad5gradients_2/Classify/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ż
'gradients_2/Classify/add_2_grad/ReshapeReshape#gradients_2/Classify/add_2_grad/Sum%gradients_2/Classify/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
%gradients_2/Classify/add_2_grad/Sum_1Sum)gradients_2/Classify/Relu_2_grad/ReluGrad7gradients_2/Classify/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
¸
)gradients_2/Classify/add_2_grad/Reshape_1Reshape%gradients_2/Classify/add_2_grad/Sum_1'gradients_2/Classify/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

0gradients_2/Classify/add_2_grad/tuple/group_depsNoOp(^gradients_2/Classify/add_2_grad/Reshape*^gradients_2/Classify/add_2_grad/Reshape_1

8gradients_2/Classify/add_2_grad/tuple/control_dependencyIdentity'gradients_2/Classify/add_2_grad/Reshape1^gradients_2/Classify/add_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_2/Classify/add_2_grad/tuple/control_dependency_1Identity)gradients_2/Classify/add_2_grad/Reshape_11^gradients_2/Classify/add_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/add_2_grad/Reshape_1*
_output_shapes	
:
Ř
)gradients_2/Classify/MatMul_2_grad/MatMulMatMul8gradients_2/Classify/add_2_grad/tuple/control_dependencyClassify/w3/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
+gradients_2/Classify/MatMul_2_grad/MatMul_1MatMulClassify/Relu_18gradients_2/Classify/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


3gradients_2/Classify/MatMul_2_grad/tuple/group_depsNoOp*^gradients_2/Classify/MatMul_2_grad/MatMul,^gradients_2/Classify/MatMul_2_grad/MatMul_1

;gradients_2/Classify/MatMul_2_grad/tuple/control_dependencyIdentity)gradients_2/Classify/MatMul_2_grad/MatMul4^gradients_2/Classify/MatMul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/Classify/MatMul_2_grad/tuple/control_dependency_1Identity+gradients_2/Classify/MatMul_2_grad/MatMul_14^gradients_2/Classify/MatMul_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/Classify/MatMul_2_grad/MatMul_1* 
_output_shapes
:

ś
)gradients_2/Classify/Relu_1_grad/ReluGradReluGrad;gradients_2/Classify/MatMul_2_grad/tuple/control_dependencyClassify/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
%gradients_2/Classify/add_1_grad/ShapeShapeClassify/MatMul_1*
T0*
out_type0*
_output_shapes
:
r
'gradients_2/Classify/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ű
5gradients_2/Classify/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_2/Classify/add_1_grad/Shape'gradients_2/Classify/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ě
#gradients_2/Classify/add_1_grad/SumSum)gradients_2/Classify/Relu_1_grad/ReluGrad5gradients_2/Classify/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ż
'gradients_2/Classify/add_1_grad/ReshapeReshape#gradients_2/Classify/add_1_grad/Sum%gradients_2/Classify/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
%gradients_2/Classify/add_1_grad/Sum_1Sum)gradients_2/Classify/Relu_1_grad/ReluGrad7gradients_2/Classify/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
¸
)gradients_2/Classify/add_1_grad/Reshape_1Reshape%gradients_2/Classify/add_1_grad/Sum_1'gradients_2/Classify/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

0gradients_2/Classify/add_1_grad/tuple/group_depsNoOp(^gradients_2/Classify/add_1_grad/Reshape*^gradients_2/Classify/add_1_grad/Reshape_1

8gradients_2/Classify/add_1_grad/tuple/control_dependencyIdentity'gradients_2/Classify/add_1_grad/Reshape1^gradients_2/Classify/add_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_2/Classify/add_1_grad/tuple/control_dependency_1Identity)gradients_2/Classify/add_1_grad/Reshape_11^gradients_2/Classify/add_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/add_1_grad/Reshape_1*
_output_shapes	
:
Ř
)gradients_2/Classify/MatMul_1_grad/MatMulMatMul8gradients_2/Classify/add_1_grad/tuple/control_dependencyClassify/w2/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
+gradients_2/Classify/MatMul_1_grad/MatMul_1MatMulClassify/Relu8gradients_2/Classify/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


3gradients_2/Classify/MatMul_1_grad/tuple/group_depsNoOp*^gradients_2/Classify/MatMul_1_grad/MatMul,^gradients_2/Classify/MatMul_1_grad/MatMul_1

;gradients_2/Classify/MatMul_1_grad/tuple/control_dependencyIdentity)gradients_2/Classify/MatMul_1_grad/MatMul4^gradients_2/Classify/MatMul_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_2/Classify/MatMul_1_grad/tuple/control_dependency_1Identity+gradients_2/Classify/MatMul_1_grad/MatMul_14^gradients_2/Classify/MatMul_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_2/Classify/MatMul_1_grad/MatMul_1* 
_output_shapes
:

˛
'gradients_2/Classify/Relu_grad/ReluGradReluGrad;gradients_2/Classify/MatMul_1_grad/tuple/control_dependencyClassify/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
#gradients_2/Classify/add_grad/ShapeShapeClassify/MatMul*
T0*
out_type0*
_output_shapes
:
p
%gradients_2/Classify/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ő
3gradients_2/Classify/add_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_2/Classify/add_grad/Shape%gradients_2/Classify/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ć
!gradients_2/Classify/add_grad/SumSum'gradients_2/Classify/Relu_grad/ReluGrad3gradients_2/Classify/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
š
%gradients_2/Classify/add_grad/ReshapeReshape!gradients_2/Classify/add_grad/Sum#gradients_2/Classify/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
#gradients_2/Classify/add_grad/Sum_1Sum'gradients_2/Classify/Relu_grad/ReluGrad5gradients_2/Classify/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
˛
'gradients_2/Classify/add_grad/Reshape_1Reshape#gradients_2/Classify/add_grad/Sum_1%gradients_2/Classify/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

.gradients_2/Classify/add_grad/tuple/group_depsNoOp&^gradients_2/Classify/add_grad/Reshape(^gradients_2/Classify/add_grad/Reshape_1

6gradients_2/Classify/add_grad/tuple/control_dependencyIdentity%gradients_2/Classify/add_grad/Reshape/^gradients_2/Classify/add_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients_2/Classify/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients_2/Classify/add_grad/tuple/control_dependency_1Identity'gradients_2/Classify/add_grad/Reshape_1/^gradients_2/Classify/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/add_grad/Reshape_1*
_output_shapes	
:
Ó
'gradients_2/Classify/MatMul_grad/MatMulMatMul6gradients_2/Classify/add_grad/tuple/control_dependencyClassify/w1/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Č
)gradients_2/Classify/MatMul_grad/MatMul_1MatMulPlaceholder6gradients_2/Classify/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	2

1gradients_2/Classify/MatMul_grad/tuple/group_depsNoOp(^gradients_2/Classify/MatMul_grad/MatMul*^gradients_2/Classify/MatMul_grad/MatMul_1

9gradients_2/Classify/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/Classify/MatMul_grad/MatMul2^gradients_2/Classify/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/Classify/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

;gradients_2/Classify/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/Classify/MatMul_grad/MatMul_12^gradients_2/Classify/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/Classify/MatMul_grad/MatMul_1*
_output_shapes
:	2

beta1_power/initial_valueConst*!
_class
loc:@Classify/bias1*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *!
_class
loc:@Classify/bias1*
	container *
shape: *
dtype0*
_output_shapes
: 
ą
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
m
beta1_power/readIdentitybeta1_power*
T0*!
_class
loc:@Classify/bias1*
_output_shapes
: 

beta2_power/initial_valueConst*!
_class
loc:@Classify/bias1*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *!
_class
loc:@Classify/bias1*
	container *
shape: *
dtype0*
_output_shapes
: 
ą
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
m
beta2_power/readIdentitybeta2_power*
T0*!
_class
loc:@Classify/bias1*
_output_shapes
: 
Ł
2Classify/w1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"2      *
_class
loc:@Classify/w1*
dtype0*
_output_shapes
:

(Classify/w1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w1*
dtype0*
_output_shapes
: 
ä
"Classify/w1/Adam/Initializer/zerosFill2Classify/w1/Adam/Initializer/zeros/shape_as_tensor(Classify/w1/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w1*
_output_shapes
:	2
Ś
Classify/w1/Adam
VariableV2*
shared_name *
_class
loc:@Classify/w1*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ę
Classify/w1/Adam/AssignAssignClassify/w1/Adam"Classify/w1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2
}
Classify/w1/Adam/readIdentityClassify/w1/Adam*
T0*
_class
loc:@Classify/w1*
_output_shapes
:	2
Ľ
4Classify/w1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"2      *
_class
loc:@Classify/w1*
dtype0*
_output_shapes
:

*Classify/w1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w1*
dtype0*
_output_shapes
: 
ę
$Classify/w1/Adam_1/Initializer/zerosFill4Classify/w1/Adam_1/Initializer/zeros/shape_as_tensor*Classify/w1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w1*
_output_shapes
:	2
¨
Classify/w1/Adam_1
VariableV2*
shared_name *
_class
loc:@Classify/w1*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Đ
Classify/w1/Adam_1/AssignAssignClassify/w1/Adam_1$Classify/w1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2

Classify/w1/Adam_1/readIdentityClassify/w1/Adam_1*
T0*
_class
loc:@Classify/w1*
_output_shapes
:	2

%Classify/bias1/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias1*
dtype0*
_output_shapes	
:
¤
Classify/bias1/Adam
VariableV2*
shared_name *!
_class
loc:@Classify/bias1*
	container *
shape:*
dtype0*
_output_shapes	
:
Ň
Classify/bias1/Adam/AssignAssignClassify/bias1/Adam%Classify/bias1/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:

Classify/bias1/Adam/readIdentityClassify/bias1/Adam*
T0*!
_class
loc:@Classify/bias1*
_output_shapes	
:

'Classify/bias1/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias1*
dtype0*
_output_shapes	
:
Ś
Classify/bias1/Adam_1
VariableV2*
shared_name *!
_class
loc:@Classify/bias1*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Classify/bias1/Adam_1/AssignAssignClassify/bias1/Adam_1'Classify/bias1/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:

Classify/bias1/Adam_1/readIdentityClassify/bias1/Adam_1*
T0*!
_class
loc:@Classify/bias1*
_output_shapes	
:
Ł
2Classify/w2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@Classify/w2*
dtype0*
_output_shapes
:

(Classify/w2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w2*
dtype0*
_output_shapes
: 
ĺ
"Classify/w2/Adam/Initializer/zerosFill2Classify/w2/Adam/Initializer/zeros/shape_as_tensor(Classify/w2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w2* 
_output_shapes
:

¨
Classify/w2/Adam
VariableV2*
shared_name *
_class
loc:@Classify/w2*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ë
Classify/w2/Adam/AssignAssignClassify/w2/Adam"Classify/w2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:

~
Classify/w2/Adam/readIdentityClassify/w2/Adam*
T0*
_class
loc:@Classify/w2* 
_output_shapes
:

Ľ
4Classify/w2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@Classify/w2*
dtype0*
_output_shapes
:

*Classify/w2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w2*
dtype0*
_output_shapes
: 
ë
$Classify/w2/Adam_1/Initializer/zerosFill4Classify/w2/Adam_1/Initializer/zeros/shape_as_tensor*Classify/w2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w2* 
_output_shapes
:

Ş
Classify/w2/Adam_1
VariableV2*
shared_name *
_class
loc:@Classify/w2*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ń
Classify/w2/Adam_1/AssignAssignClassify/w2/Adam_1$Classify/w2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:


Classify/w2/Adam_1/readIdentityClassify/w2/Adam_1*
T0*
_class
loc:@Classify/w2* 
_output_shapes
:


%Classify/bias2/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias2*
dtype0*
_output_shapes	
:
¤
Classify/bias2/Adam
VariableV2*
shared_name *!
_class
loc:@Classify/bias2*
	container *
shape:*
dtype0*
_output_shapes	
:
Ň
Classify/bias2/Adam/AssignAssignClassify/bias2/Adam%Classify/bias2/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:

Classify/bias2/Adam/readIdentityClassify/bias2/Adam*
T0*!
_class
loc:@Classify/bias2*
_output_shapes	
:

'Classify/bias2/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias2*
dtype0*
_output_shapes	
:
Ś
Classify/bias2/Adam_1
VariableV2*
shared_name *!
_class
loc:@Classify/bias2*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Classify/bias2/Adam_1/AssignAssignClassify/bias2/Adam_1'Classify/bias2/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:

Classify/bias2/Adam_1/readIdentityClassify/bias2/Adam_1*
T0*!
_class
loc:@Classify/bias2*
_output_shapes	
:
Ł
2Classify/w3/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@Classify/w3*
dtype0*
_output_shapes
:

(Classify/w3/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w3*
dtype0*
_output_shapes
: 
ĺ
"Classify/w3/Adam/Initializer/zerosFill2Classify/w3/Adam/Initializer/zeros/shape_as_tensor(Classify/w3/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w3* 
_output_shapes
:

¨
Classify/w3/Adam
VariableV2*
shared_name *
_class
loc:@Classify/w3*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ë
Classify/w3/Adam/AssignAssignClassify/w3/Adam"Classify/w3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:

~
Classify/w3/Adam/readIdentityClassify/w3/Adam*
T0*
_class
loc:@Classify/w3* 
_output_shapes
:

Ľ
4Classify/w3/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@Classify/w3*
dtype0*
_output_shapes
:

*Classify/w3/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Classify/w3*
dtype0*
_output_shapes
: 
ë
$Classify/w3/Adam_1/Initializer/zerosFill4Classify/w3/Adam_1/Initializer/zeros/shape_as_tensor*Classify/w3/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Classify/w3* 
_output_shapes
:

Ş
Classify/w3/Adam_1
VariableV2*
shared_name *
_class
loc:@Classify/w3*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ń
Classify/w3/Adam_1/AssignAssignClassify/w3/Adam_1$Classify/w3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:


Classify/w3/Adam_1/readIdentityClassify/w3/Adam_1*
T0*
_class
loc:@Classify/w3* 
_output_shapes
:


%Classify/bias3/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias3*
dtype0*
_output_shapes	
:
¤
Classify/bias3/Adam
VariableV2*
shared_name *!
_class
loc:@Classify/bias3*
	container *
shape:*
dtype0*
_output_shapes	
:
Ň
Classify/bias3/Adam/AssignAssignClassify/bias3/Adam%Classify/bias3/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:

Classify/bias3/Adam/readIdentityClassify/bias3/Adam*
T0*!
_class
loc:@Classify/bias3*
_output_shapes	
:

'Classify/bias3/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@Classify/bias3*
dtype0*
_output_shapes	
:
Ś
Classify/bias3/Adam_1
VariableV2*
shared_name *!
_class
loc:@Classify/bias3*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Classify/bias3/Adam_1/AssignAssignClassify/bias3/Adam_1'Classify/bias3/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:

Classify/bias3/Adam_1/readIdentityClassify/bias3/Adam_1*
T0*!
_class
loc:@Classify/bias3*
_output_shapes	
:
§
4Classify/w4_1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_1*
dtype0*
_output_shapes
:

*Classify/w4_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_1*
dtype0*
_output_shapes
: 
ě
$Classify/w4_1/Adam/Initializer/zerosFill4Classify/w4_1/Adam/Initializer/zeros/shape_as_tensor*Classify/w4_1/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	
Ş
Classify/w4_1/Adam
VariableV2*
shared_name * 
_class
loc:@Classify/w4_1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ň
Classify/w4_1/Adam/AssignAssignClassify/w4_1/Adam$Classify/w4_1/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	

Classify/w4_1/Adam/readIdentityClassify/w4_1/Adam*
T0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	
Š
6Classify/w4_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_1*
dtype0*
_output_shapes
:

,Classify/w4_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_1*
dtype0*
_output_shapes
: 
ň
&Classify/w4_1/Adam_1/Initializer/zerosFill6Classify/w4_1/Adam_1/Initializer/zeros/shape_as_tensor,Classify/w4_1/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	
Ź
Classify/w4_1/Adam_1
VariableV2*
shared_name * 
_class
loc:@Classify/w4_1*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ř
Classify/w4_1/Adam_1/AssignAssignClassify/w4_1/Adam_1&Classify/w4_1/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	

Classify/w4_1/Adam_1/readIdentityClassify/w4_1/Adam_1*
T0* 
_class
loc:@Classify/w4_1*
_output_shapes
:	

'Classify/bias4_1/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_1*
dtype0*
_output_shapes
:
Ś
Classify/bias4_1/Adam
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_1*
	container *
shape:*
dtype0*
_output_shapes
:
Ů
Classify/bias4_1/Adam/AssignAssignClassify/bias4_1/Adam'Classify/bias4_1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:

Classify/bias4_1/Adam/readIdentityClassify/bias4_1/Adam*
T0*#
_class
loc:@Classify/bias4_1*
_output_shapes
:

)Classify/bias4_1/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_1*
dtype0*
_output_shapes
:
¨
Classify/bias4_1/Adam_1
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_1*
	container *
shape:*
dtype0*
_output_shapes
:
ß
Classify/bias4_1/Adam_1/AssignAssignClassify/bias4_1/Adam_1)Classify/bias4_1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:

Classify/bias4_1/Adam_1/readIdentityClassify/bias4_1/Adam_1*
T0*#
_class
loc:@Classify/bias4_1*
_output_shapes
:
§
4Classify/w4_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_2*
dtype0*
_output_shapes
:

*Classify/w4_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_2*
dtype0*
_output_shapes
: 
ě
$Classify/w4_2/Adam/Initializer/zerosFill4Classify/w4_2/Adam/Initializer/zeros/shape_as_tensor*Classify/w4_2/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	
Ş
Classify/w4_2/Adam
VariableV2*
shared_name * 
_class
loc:@Classify/w4_2*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ň
Classify/w4_2/Adam/AssignAssignClassify/w4_2/Adam$Classify/w4_2/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	

Classify/w4_2/Adam/readIdentityClassify/w4_2/Adam*
T0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	
Š
6Classify/w4_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_2*
dtype0*
_output_shapes
:

,Classify/w4_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_2*
dtype0*
_output_shapes
: 
ň
&Classify/w4_2/Adam_1/Initializer/zerosFill6Classify/w4_2/Adam_1/Initializer/zeros/shape_as_tensor,Classify/w4_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	
Ź
Classify/w4_2/Adam_1
VariableV2*
shared_name * 
_class
loc:@Classify/w4_2*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ř
Classify/w4_2/Adam_1/AssignAssignClassify/w4_2/Adam_1&Classify/w4_2/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	

Classify/w4_2/Adam_1/readIdentityClassify/w4_2/Adam_1*
T0* 
_class
loc:@Classify/w4_2*
_output_shapes
:	

'Classify/bias4_2/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_2*
dtype0*
_output_shapes
:
Ś
Classify/bias4_2/Adam
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_2*
	container *
shape:*
dtype0*
_output_shapes
:
Ů
Classify/bias4_2/Adam/AssignAssignClassify/bias4_2/Adam'Classify/bias4_2/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:

Classify/bias4_2/Adam/readIdentityClassify/bias4_2/Adam*
T0*#
_class
loc:@Classify/bias4_2*
_output_shapes
:

)Classify/bias4_2/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_2*
dtype0*
_output_shapes
:
¨
Classify/bias4_2/Adam_1
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_2*
	container *
shape:*
dtype0*
_output_shapes
:
ß
Classify/bias4_2/Adam_1/AssignAssignClassify/bias4_2/Adam_1)Classify/bias4_2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:

Classify/bias4_2/Adam_1/readIdentityClassify/bias4_2/Adam_1*
T0*#
_class
loc:@Classify/bias4_2*
_output_shapes
:
§
4Classify/w4_3/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_3*
dtype0*
_output_shapes
:

*Classify/w4_3/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_3*
dtype0*
_output_shapes
: 
ě
$Classify/w4_3/Adam/Initializer/zerosFill4Classify/w4_3/Adam/Initializer/zeros/shape_as_tensor*Classify/w4_3/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	
Ş
Classify/w4_3/Adam
VariableV2*
shared_name * 
_class
loc:@Classify/w4_3*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ň
Classify/w4_3/Adam/AssignAssignClassify/w4_3/Adam$Classify/w4_3/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	

Classify/w4_3/Adam/readIdentityClassify/w4_3/Adam*
T0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	
Š
6Classify/w4_3/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      * 
_class
loc:@Classify/w4_3*
dtype0*
_output_shapes
:

,Classify/w4_3/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@Classify/w4_3*
dtype0*
_output_shapes
: 
ň
&Classify/w4_3/Adam_1/Initializer/zerosFill6Classify/w4_3/Adam_1/Initializer/zeros/shape_as_tensor,Classify/w4_3/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	
Ź
Classify/w4_3/Adam_1
VariableV2*
shared_name * 
_class
loc:@Classify/w4_3*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ř
Classify/w4_3/Adam_1/AssignAssignClassify/w4_3/Adam_1&Classify/w4_3/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	

Classify/w4_3/Adam_1/readIdentityClassify/w4_3/Adam_1*
T0* 
_class
loc:@Classify/w4_3*
_output_shapes
:	

'Classify/bias4_3/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_3*
dtype0*
_output_shapes
:
Ś
Classify/bias4_3/Adam
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_3*
	container *
shape:*
dtype0*
_output_shapes
:
Ů
Classify/bias4_3/Adam/AssignAssignClassify/bias4_3/Adam'Classify/bias4_3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:

Classify/bias4_3/Adam/readIdentityClassify/bias4_3/Adam*
T0*#
_class
loc:@Classify/bias4_3*
_output_shapes
:

)Classify/bias4_3/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@Classify/bias4_3*
dtype0*
_output_shapes
:
¨
Classify/bias4_3/Adam_1
VariableV2*
shared_name *#
_class
loc:@Classify/bias4_3*
	container *
shape:*
dtype0*
_output_shapes
:
ß
Classify/bias4_3/Adam_1/AssignAssignClassify/bias4_3/Adam_1)Classify/bias4_3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:

Classify/bias4_3/Adam_1/readIdentityClassify/bias4_3/Adam_1*
T0*#
_class
loc:@Classify/bias4_3*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
í
!Adam/update_Classify/w1/ApplyAdam	ApplyAdamClassify/w1Classify/w1/AdamClassify/w1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients_2/Classify/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Classify/w1*
use_nesterov( *
_output_shapes
:	2
ő
$Adam/update_Classify/bias1/ApplyAdam	ApplyAdamClassify/bias1Classify/bias1/AdamClassify/bias1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients_2/Classify/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Classify/bias1*
use_nesterov( *
_output_shapes	
:
đ
!Adam/update_Classify/w2/ApplyAdam	ApplyAdamClassify/w2Classify/w2/AdamClassify/w2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients_2/Classify/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Classify/w2*
use_nesterov( * 
_output_shapes
:

÷
$Adam/update_Classify/bias2/ApplyAdam	ApplyAdamClassify/bias2Classify/bias2/AdamClassify/bias2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients_2/Classify/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Classify/bias2*
use_nesterov( *
_output_shapes	
:
đ
!Adam/update_Classify/w3/ApplyAdam	ApplyAdamClassify/w3Classify/w3/AdamClassify/w3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients_2/Classify/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Classify/w3*
use_nesterov( * 
_output_shapes
:

÷
$Adam/update_Classify/bias3/ApplyAdam	ApplyAdamClassify/bias3Classify/bias3/AdamClassify/bias3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients_2/Classify/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Classify/bias3*
use_nesterov( *
_output_shapes	
:
ů
#Adam/update_Classify/w4_1/ApplyAdam	ApplyAdamClassify/w4_1Classify/w4_1/AdamClassify/w4_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients_2/Classify/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@Classify/w4_1*
use_nesterov( *
_output_shapes
:	

&Adam/update_Classify/bias4_1/ApplyAdam	ApplyAdamClassify/bias4_1Classify/bias4_1/AdamClassify/bias4_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients_2/Classify/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@Classify/bias4_1*
use_nesterov( *
_output_shapes
:
ů
#Adam/update_Classify/w4_2/ApplyAdam	ApplyAdamClassify/w4_2Classify/w4_2/AdamClassify/w4_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients_2/Classify/MatMul_4_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@Classify/w4_2*
use_nesterov( *
_output_shapes
:	

&Adam/update_Classify/bias4_2/ApplyAdam	ApplyAdamClassify/bias4_2Classify/bias4_2/AdamClassify/bias4_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients_2/Classify/add_4_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@Classify/bias4_2*
use_nesterov( *
_output_shapes
:
ů
#Adam/update_Classify/w4_3/ApplyAdam	ApplyAdamClassify/w4_3Classify/w4_3/AdamClassify/w4_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients_2/Classify/MatMul_5_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@Classify/w4_3*
use_nesterov( *
_output_shapes
:	

&Adam/update_Classify/bias4_3/ApplyAdam	ApplyAdamClassify/bias4_3Classify/bias4_3/AdamClassify/bias4_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients_2/Classify/add_5_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@Classify/bias4_3*
use_nesterov( *
_output_shapes
:
ż
Adam/mulMulbeta1_power/read
Adam/beta1%^Adam/update_Classify/bias1/ApplyAdam%^Adam/update_Classify/bias2/ApplyAdam%^Adam/update_Classify/bias3/ApplyAdam'^Adam/update_Classify/bias4_1/ApplyAdam'^Adam/update_Classify/bias4_2/ApplyAdam'^Adam/update_Classify/bias4_3/ApplyAdam"^Adam/update_Classify/w1/ApplyAdam"^Adam/update_Classify/w2/ApplyAdam"^Adam/update_Classify/w3/ApplyAdam$^Adam/update_Classify/w4_1/ApplyAdam$^Adam/update_Classify/w4_2/ApplyAdam$^Adam/update_Classify/w4_3/ApplyAdam*
T0*!
_class
loc:@Classify/bias1*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
Á

Adam/mul_1Mulbeta2_power/read
Adam/beta2%^Adam/update_Classify/bias1/ApplyAdam%^Adam/update_Classify/bias2/ApplyAdam%^Adam/update_Classify/bias3/ApplyAdam'^Adam/update_Classify/bias4_1/ApplyAdam'^Adam/update_Classify/bias4_2/ApplyAdam'^Adam/update_Classify/bias4_3/ApplyAdam"^Adam/update_Classify/w1/ApplyAdam"^Adam/update_Classify/w2/ApplyAdam"^Adam/update_Classify/w3/ApplyAdam$^Adam/update_Classify/w4_1/ApplyAdam$^Adam/update_Classify/w4_2/ApplyAdam$^Adam/update_Classify/w4_3/ApplyAdam*
T0*!
_class
loc:@Classify/bias1*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
ř
AdamNoOp^Adam/Assign^Adam/Assign_1%^Adam/update_Classify/bias1/ApplyAdam%^Adam/update_Classify/bias2/ApplyAdam%^Adam/update_Classify/bias3/ApplyAdam'^Adam/update_Classify/bias4_1/ApplyAdam'^Adam/update_Classify/bias4_2/ApplyAdam'^Adam/update_Classify/bias4_3/ApplyAdam"^Adam/update_Classify/w1/ApplyAdam"^Adam/update_Classify/w2/ApplyAdam"^Adam/update_Classify/w3/ApplyAdam$^Adam/update_Classify/w4_1/ApplyAdam$^Adam/update_Classify/w4_2/ApplyAdam$^Adam/update_Classify/w4_3/ApplyAdam
T
gradients_3/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_3/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
t
#gradients_3/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients_3/Mean_grad/ReshapeReshapegradients_3/Fill#gradients_3/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
a
gradients_3/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
˘
gradients_3/Mean_grad/TileTilegradients_3/Mean_grad/Reshapegradients_3/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_3/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
`
gradients_3/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients_3/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients_3/Mean_grad/ProdProdgradients_3/Mean_grad/Shape_1gradients_3/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients_3/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients_3/Mean_grad/Prod_1Prodgradients_3/Mean_grad/Shape_2gradients_3/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
a
gradients_3/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_3/Mean_grad/MaximumMaximumgradients_3/Mean_grad/Prod_1gradients_3/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients_3/Mean_grad/floordivFloorDivgradients_3/Mean_grad/Prodgradients_3/Mean_grad/Maximum*
T0*
_output_shapes
: 

gradients_3/Mean_grad/CastCastgradients_3/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_3/Mean_grad/truedivRealDivgradients_3/Mean_grad/Tilegradients_3/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_3/Square_grad/ConstConst^gradients_3/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
z
gradients_3/Square_grad/MulMulsub_2gradients_3/Square_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_3/Square_grad/Mul_1Mulgradients_3/Mean_grad/truedivgradients_3/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients_3/sub_2_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
m
gradients_3/sub_2_grad/Shape_1ShapeRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_3/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_2_grad/Shapegradients_3/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
gradients_3/sub_2_grad/SumSumgradients_3/Square_grad/Mul_1,gradients_3/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_3/sub_2_grad/ReshapeReshapegradients_3/sub_2_grad/Sumgradients_3/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients_3/sub_2_grad/Sum_1Sumgradients_3/Square_grad/Mul_1.gradients_3/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_3/sub_2_grad/NegNeggradients_3/sub_2_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_3/sub_2_grad/Reshape_1Reshapegradients_3/sub_2_grad/Neggradients_3/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_3/sub_2_grad/tuple/group_depsNoOp^gradients_3/sub_2_grad/Reshape!^gradients_3/sub_2_grad/Reshape_1
ę
/gradients_3/sub_2_grad/tuple/control_dependencyIdentitygradients_3/sub_2_grad/Reshape(^gradients_3/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_3/sub_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_3/sub_2_grad/tuple/control_dependency_1Identity gradients_3/sub_2_grad/Reshape_1(^gradients_3/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_3/sub_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
,gradients_3/Regress/Sigmoid_grad/SigmoidGradSigmoidGradRegress/Sigmoid1gradients_3/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_3/Regress/add_3_grad/ShapeShapeRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
p
&gradients_3/Regress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_3/Regress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_3/Regress/add_3_grad/Shape&gradients_3/Regress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Í
"gradients_3/Regress/add_3_grad/SumSum,gradients_3/Regress/Sigmoid_grad/SigmoidGrad4gradients_3/Regress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ť
&gradients_3/Regress/add_3_grad/ReshapeReshape"gradients_3/Regress/add_3_grad/Sum$gradients_3/Regress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
$gradients_3/Regress/add_3_grad/Sum_1Sum,gradients_3/Regress/Sigmoid_grad/SigmoidGrad6gradients_3/Regress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
´
(gradients_3/Regress/add_3_grad/Reshape_1Reshape$gradients_3/Regress/add_3_grad/Sum_1&gradients_3/Regress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

/gradients_3/Regress/add_3_grad/tuple/group_depsNoOp'^gradients_3/Regress/add_3_grad/Reshape)^gradients_3/Regress/add_3_grad/Reshape_1

7gradients_3/Regress/add_3_grad/tuple/control_dependencyIdentity&gradients_3/Regress/add_3_grad/Reshape0^gradients_3/Regress/add_3_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_3/Regress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_3/Regress/add_3_grad/tuple/control_dependency_1Identity(gradients_3/Regress/add_3_grad/Reshape_10^gradients_3/Regress/add_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/add_3_grad/Reshape_1*
_output_shapes
:
Ů
(gradients_3/Regress/MatMul_3_grad/MatMulMatMul7gradients_3/Regress/add_3_grad/tuple/control_dependencyRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
*gradients_3/Regress/MatMul_3_grad/MatMul_1MatMulRegress/Relu_27gradients_3/Regress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

2gradients_3/Regress/MatMul_3_grad/tuple/group_depsNoOp)^gradients_3/Regress/MatMul_3_grad/MatMul+^gradients_3/Regress/MatMul_3_grad/MatMul_1

:gradients_3/Regress/MatMul_3_grad/tuple/control_dependencyIdentity(gradients_3/Regress/MatMul_3_grad/MatMul3^gradients_3/Regress/MatMul_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_3/Regress/MatMul_3_grad/tuple/control_dependency_1Identity*gradients_3/Regress/MatMul_3_grad/MatMul_13^gradients_3/Regress/MatMul_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_3/Regress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
ł
(gradients_3/Regress/Relu_2_grad/ReluGradReluGrad:gradients_3/Regress/MatMul_3_grad/tuple/control_dependencyRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_3/Regress/add_2_grad/ShapeShapeRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
q
&gradients_3/Regress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_3/Regress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_3/Regress/add_2_grad/Shape&gradients_3/Regress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_3/Regress/add_2_grad/SumSum(gradients_3/Regress/Relu_2_grad/ReluGrad4gradients_3/Regress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_3/Regress/add_2_grad/ReshapeReshape"gradients_3/Regress/add_2_grad/Sum$gradients_3/Regress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_3/Regress/add_2_grad/Sum_1Sum(gradients_3/Regress/Relu_2_grad/ReluGrad6gradients_3/Regress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_3/Regress/add_2_grad/Reshape_1Reshape$gradients_3/Regress/add_2_grad/Sum_1&gradients_3/Regress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_3/Regress/add_2_grad/tuple/group_depsNoOp'^gradients_3/Regress/add_2_grad/Reshape)^gradients_3/Regress/add_2_grad/Reshape_1

7gradients_3/Regress/add_2_grad/tuple/control_dependencyIdentity&gradients_3/Regress/add_2_grad/Reshape0^gradients_3/Regress/add_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_3/Regress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_3/Regress/add_2_grad/tuple/control_dependency_1Identity(gradients_3/Regress/add_2_grad/Reshape_10^gradients_3/Regress/add_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/add_2_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_3/Regress/MatMul_2_grad/MatMulMatMul7gradients_3/Regress/add_2_grad/tuple/control_dependencyRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
*gradients_3/Regress/MatMul_2_grad/MatMul_1MatMulRegress/Relu_17gradients_3/Regress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_3/Regress/MatMul_2_grad/tuple/group_depsNoOp)^gradients_3/Regress/MatMul_2_grad/MatMul+^gradients_3/Regress/MatMul_2_grad/MatMul_1

:gradients_3/Regress/MatMul_2_grad/tuple/control_dependencyIdentity(gradients_3/Regress/MatMul_2_grad/MatMul3^gradients_3/Regress/MatMul_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_3/Regress/MatMul_2_grad/tuple/control_dependency_1Identity*gradients_3/Regress/MatMul_2_grad/MatMul_13^gradients_3/Regress/MatMul_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_3/Regress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

ł
(gradients_3/Regress/Relu_1_grad/ReluGradReluGrad:gradients_3/Regress/MatMul_2_grad/tuple/control_dependencyRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_3/Regress/add_1_grad/ShapeShapeRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
q
&gradients_3/Regress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_3/Regress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_3/Regress/add_1_grad/Shape&gradients_3/Regress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_3/Regress/add_1_grad/SumSum(gradients_3/Regress/Relu_1_grad/ReluGrad4gradients_3/Regress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_3/Regress/add_1_grad/ReshapeReshape"gradients_3/Regress/add_1_grad/Sum$gradients_3/Regress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_3/Regress/add_1_grad/Sum_1Sum(gradients_3/Regress/Relu_1_grad/ReluGrad6gradients_3/Regress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_3/Regress/add_1_grad/Reshape_1Reshape$gradients_3/Regress/add_1_grad/Sum_1&gradients_3/Regress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_3/Regress/add_1_grad/tuple/group_depsNoOp'^gradients_3/Regress/add_1_grad/Reshape)^gradients_3/Regress/add_1_grad/Reshape_1

7gradients_3/Regress/add_1_grad/tuple/control_dependencyIdentity&gradients_3/Regress/add_1_grad/Reshape0^gradients_3/Regress/add_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_3/Regress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_3/Regress/add_1_grad/tuple/control_dependency_1Identity(gradients_3/Regress/add_1_grad/Reshape_10^gradients_3/Regress/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/add_1_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_3/Regress/MatMul_1_grad/MatMulMatMul7gradients_3/Regress/add_1_grad/tuple/control_dependencyRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
*gradients_3/Regress/MatMul_1_grad/MatMul_1MatMulRegress/Relu7gradients_3/Regress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_3/Regress/MatMul_1_grad/tuple/group_depsNoOp)^gradients_3/Regress/MatMul_1_grad/MatMul+^gradients_3/Regress/MatMul_1_grad/MatMul_1

:gradients_3/Regress/MatMul_1_grad/tuple/control_dependencyIdentity(gradients_3/Regress/MatMul_1_grad/MatMul3^gradients_3/Regress/MatMul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_3/Regress/MatMul_1_grad/tuple/control_dependency_1Identity*gradients_3/Regress/MatMul_1_grad/MatMul_13^gradients_3/Regress/MatMul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_3/Regress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ż
&gradients_3/Regress/Relu_grad/ReluGradReluGrad:gradients_3/Regress/MatMul_1_grad/tuple/control_dependencyRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
"gradients_3/Regress/add_grad/ShapeShapeRegress/MatMul*
T0*
out_type0*
_output_shapes
:
o
$gradients_3/Regress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ň
2gradients_3/Regress/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_3/Regress/add_grad/Shape$gradients_3/Regress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ă
 gradients_3/Regress/add_grad/SumSum&gradients_3/Regress/Relu_grad/ReluGrad2gradients_3/Regress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ś
$gradients_3/Regress/add_grad/ReshapeReshape gradients_3/Regress/add_grad/Sum"gradients_3/Regress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
"gradients_3/Regress/add_grad/Sum_1Sum&gradients_3/Regress/Relu_grad/ReluGrad4gradients_3/Regress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ż
&gradients_3/Regress/add_grad/Reshape_1Reshape"gradients_3/Regress/add_grad/Sum_1$gradients_3/Regress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

-gradients_3/Regress/add_grad/tuple/group_depsNoOp%^gradients_3/Regress/add_grad/Reshape'^gradients_3/Regress/add_grad/Reshape_1

5gradients_3/Regress/add_grad/tuple/control_dependencyIdentity$gradients_3/Regress/add_grad/Reshape.^gradients_3/Regress/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_3/Regress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
7gradients_3/Regress/add_grad/tuple/control_dependency_1Identity&gradients_3/Regress/add_grad/Reshape_1.^gradients_3/Regress/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_3/Regress/add_grad/Reshape_1*
_output_shapes	
:
Ô
&gradients_3/Regress/MatMul_grad/MatMulMatMul5gradients_3/Regress/add_grad/tuple/control_dependencyRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Č
(gradients_3/Regress/MatMul_grad/MatMul_1MatMulPlaceholder_15gradients_3/Regress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	2

0gradients_3/Regress/MatMul_grad/tuple/group_depsNoOp'^gradients_3/Regress/MatMul_grad/MatMul)^gradients_3/Regress/MatMul_grad/MatMul_1

8gradients_3/Regress/MatMul_grad/tuple/control_dependencyIdentity&gradients_3/Regress/MatMul_grad/MatMul1^gradients_3/Regress/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_3/Regress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

:gradients_3/Regress/MatMul_grad/tuple/control_dependency_1Identity(gradients_3/Regress/MatMul_grad/MatMul_11^gradients_3/Regress/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_3/Regress/MatMul_grad/MatMul_1*
_output_shapes
:	2

beta1_power_1/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta1_power_1/readIdentitybeta1_power_1*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 

beta2_power_1/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta2_power_1/readIdentitybeta2_power_1*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Š
5Regress/w1_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

+Regress/w1_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
đ
%Regress/w1_reg/Adam/Initializer/zerosFill5Regress/w1_reg/Adam/Initializer/zeros/shape_as_tensor+Regress/w1_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ź
Regress/w1_reg/Adam
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ö
Regress/w1_reg/Adam/AssignAssignRegress/w1_reg/Adam%Regress/w1_reg/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam/readIdentityRegress/w1_reg/Adam*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ť
7Regress/w1_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

-Regress/w1_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
ö
'Regress/w1_reg/Adam_1/Initializer/zerosFill7Regress/w1_reg/Adam_1/Initializer/zeros/shape_as_tensor-Regress/w1_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ž
Regress/w1_reg/Adam_1
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ü
Regress/w1_reg/Adam_1/AssignAssignRegress/w1_reg/Adam_1'Regress/w1_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam_1/readIdentityRegress/w1_reg/Adam_1*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2

(Regress/bias1_reg/Adam/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ş
Regress/bias1_reg/Adam
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ţ
Regress/bias1_reg/Adam/AssignAssignRegress/bias1_reg/Adam(Regress/bias1_reg/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam/readIdentityRegress/bias1_reg/Adam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:

*Regress/bias1_reg/Adam_1/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias1_reg/Adam_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias1_reg/Adam_1/AssignAssignRegress/bias1_reg/Adam_1*Regress/bias1_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam_1/readIdentityRegress/bias1_reg/Adam_1*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:
Š
5Regress/w2_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

+Regress/w2_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
ń
%Regress/w2_reg/Adam/Initializer/zerosFill5Regress/w2_reg/Adam/Initializer/zeros/shape_as_tensor+Regress/w2_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Ž
Regress/w2_reg/Adam
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

×
Regress/w2_reg/Adam/AssignAssignRegress/w2_reg/Adam%Regress/w2_reg/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam/readIdentityRegress/w2_reg/Adam*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Ť
7Regress/w2_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

-Regress/w2_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w2_reg/Adam_1/Initializer/zerosFill7Regress/w2_reg/Adam_1/Initializer/zeros/shape_as_tensor-Regress/w2_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

°
Regress/w2_reg/Adam_1
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w2_reg/Adam_1/AssignAssignRegress/w2_reg/Adam_1'Regress/w2_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam_1/readIdentityRegress/w2_reg/Adam_1*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:


(Regress/bias2_reg/Adam/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ş
Regress/bias2_reg/Adam
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ţ
Regress/bias2_reg/Adam/AssignAssignRegress/bias2_reg/Adam(Regress/bias2_reg/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam/readIdentityRegress/bias2_reg/Adam*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:

*Regress/bias2_reg/Adam_1/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias2_reg/Adam_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias2_reg/Adam_1/AssignAssignRegress/bias2_reg/Adam_1*Regress/bias2_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam_1/readIdentityRegress/bias2_reg/Adam_1*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:
Š
5Regress/w3_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

+Regress/w3_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
ń
%Regress/w3_reg/Adam/Initializer/zerosFill5Regress/w3_reg/Adam/Initializer/zeros/shape_as_tensor+Regress/w3_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Ž
Regress/w3_reg/Adam
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

×
Regress/w3_reg/Adam/AssignAssignRegress/w3_reg/Adam%Regress/w3_reg/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam/readIdentityRegress/w3_reg/Adam*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Ť
7Regress/w3_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

-Regress/w3_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w3_reg/Adam_1/Initializer/zerosFill7Regress/w3_reg/Adam_1/Initializer/zeros/shape_as_tensor-Regress/w3_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

°
Regress/w3_reg/Adam_1
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w3_reg/Adam_1/AssignAssignRegress/w3_reg/Adam_1'Regress/w3_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam_1/readIdentityRegress/w3_reg/Adam_1*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:


(Regress/bias3_reg/Adam/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ş
Regress/bias3_reg/Adam
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
Ţ
Regress/bias3_reg/Adam/AssignAssignRegress/bias3_reg/Adam(Regress/bias3_reg/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam/readIdentityRegress/bias3_reg/Adam*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:

*Regress/bias3_reg/Adam_1/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias3_reg/Adam_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias3_reg/Adam_1/AssignAssignRegress/bias3_reg/Adam_1*Regress/bias3_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam_1/readIdentityRegress/bias3_reg/Adam_1*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:

%Regress/w4_reg/Adam/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ź
Regress/w4_reg/Adam
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ö
Regress/w4_reg/Adam/AssignAssignRegress/w4_reg/Adam%Regress/w4_reg/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam/readIdentityRegress/w4_reg/Adam*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	
Ą
'Regress/w4_reg/Adam_1/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ž
Regress/w4_reg/Adam_1
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
Regress/w4_reg/Adam_1/AssignAssignRegress/w4_reg/Adam_1'Regress/w4_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam_1/readIdentityRegress/w4_reg/Adam_1*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	

(Regress/bias4_reg/Adam/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
¨
Regress/bias4_reg/Adam
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
Ý
Regress/bias4_reg/Adam/AssignAssignRegress/bias4_reg/Adam(Regress/bias4_reg/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam/readIdentityRegress/bias4_reg/Adam*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:

*Regress/bias4_reg/Adam_1/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
Ş
Regress/bias4_reg/Adam_1
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ă
Regress/bias4_reg/Adam_1/AssignAssignRegress/bias4_reg/Adam_1*Regress/bias4_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam_1/readIdentityRegress/bias4_reg/Adam_1*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:
Y
Adam_1/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

&Adam_1/update_Regress/w1_reg/ApplyAdam	ApplyAdamRegress/w1_regRegress/w1_reg/AdamRegress/w1_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon:gradients_3/Regress/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w1_reg*
use_nesterov( *
_output_shapes
:	2

)Adam_1/update_Regress/bias1_reg/ApplyAdam	ApplyAdamRegress/bias1_regRegress/bias1_reg/AdamRegress/bias1_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon7gradients_3/Regress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
use_nesterov( *
_output_shapes	
:

&Adam_1/update_Regress/w2_reg/ApplyAdam	ApplyAdamRegress/w2_regRegress/w2_reg/AdamRegress/w2_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_3/Regress/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w2_reg*
use_nesterov( * 
_output_shapes
:


)Adam_1/update_Regress/bias2_reg/ApplyAdam	ApplyAdamRegress/bias2_regRegress/bias2_reg/AdamRegress/bias2_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon9gradients_3/Regress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias2_reg*
use_nesterov( *
_output_shapes	
:

&Adam_1/update_Regress/w3_reg/ApplyAdam	ApplyAdamRegress/w3_regRegress/w3_reg/AdamRegress/w3_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_3/Regress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w3_reg*
use_nesterov( * 
_output_shapes
:


)Adam_1/update_Regress/bias3_reg/ApplyAdam	ApplyAdamRegress/bias3_regRegress/bias3_reg/AdamRegress/bias3_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon9gradients_3/Regress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias3_reg*
use_nesterov( *
_output_shapes	
:

&Adam_1/update_Regress/w4_reg/ApplyAdam	ApplyAdamRegress/w4_regRegress/w4_reg/AdamRegress/w4_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_3/Regress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w4_reg*
use_nesterov( *
_output_shapes
:	

)Adam_1/update_Regress/bias4_reg/ApplyAdam	ApplyAdamRegress/bias4_regRegress/bias4_reg/AdamRegress/bias4_reg/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon9gradients_3/Regress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias4_reg*
use_nesterov( *
_output_shapes
:
Î

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1*^Adam_1/update_Regress/bias1_reg/ApplyAdam*^Adam_1/update_Regress/bias2_reg/ApplyAdam*^Adam_1/update_Regress/bias3_reg/ApplyAdam*^Adam_1/update_Regress/bias4_reg/ApplyAdam'^Adam_1/update_Regress/w1_reg/ApplyAdam'^Adam_1/update_Regress/w2_reg/ApplyAdam'^Adam_1/update_Regress/w3_reg/ApplyAdam'^Adam_1/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
˘
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Đ
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2*^Adam_1/update_Regress/bias1_reg/ApplyAdam*^Adam_1/update_Regress/bias2_reg/ApplyAdam*^Adam_1/update_Regress/bias3_reg/ApplyAdam*^Adam_1/update_Regress/bias4_reg/ApplyAdam'^Adam_1/update_Regress/w1_reg/ApplyAdam'^Adam_1/update_Regress/w2_reg/ApplyAdam'^Adam_1/update_Regress/w3_reg/ApplyAdam'^Adam_1/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Ś
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1*^Adam_1/update_Regress/bias1_reg/ApplyAdam*^Adam_1/update_Regress/bias2_reg/ApplyAdam*^Adam_1/update_Regress/bias3_reg/ApplyAdam*^Adam_1/update_Regress/bias4_reg/ApplyAdam'^Adam_1/update_Regress/w1_reg/ApplyAdam'^Adam_1/update_Regress/w2_reg/ApplyAdam'^Adam_1/update_Regress/w3_reg/ApplyAdam'^Adam_1/update_Regress/w4_reg/ApplyAdam
T
gradients_4/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_4/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
C
(gradients_4/add_20_grad/tuple/group_depsNoOp^gradients_4/Fill
ż
0gradients_4/add_20_grad/tuple/control_dependencyIdentitygradients_4/Fill)^gradients_4/add_20_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
Á
2gradients_4/add_20_grad/tuple/control_dependency_1Identitygradients_4/Fill)^gradients_4/add_20_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
t
#gradients_4/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ś
gradients_4/Mean_grad/ReshapeReshape0gradients_4/add_20_grad/tuple/control_dependency#gradients_4/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
a
gradients_4/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
˘
gradients_4/Mean_grad/TileTilegradients_4/Mean_grad/Reshapegradients_4/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_4/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
`
gradients_4/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients_4/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients_4/Mean_grad/ProdProdgradients_4/Mean_grad/Shape_1gradients_4/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients_4/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients_4/Mean_grad/Prod_1Prodgradients_4/Mean_grad/Shape_2gradients_4/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
a
gradients_4/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_4/Mean_grad/MaximumMaximumgradients_4/Mean_grad/Prod_1gradients_4/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients_4/Mean_grad/floordivFloorDivgradients_4/Mean_grad/Prodgradients_4/Mean_grad/Maximum*
T0*
_output_shapes
: 

gradients_4/Mean_grad/CastCastgradients_4/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_4/Mean_grad/truedivRealDivgradients_4/Mean_grad/Tilegradients_4/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
'gradients_4/add_8_grad/tuple/group_depsNoOp3^gradients_4/add_20_grad/tuple/control_dependency_1
ß
/gradients_4/add_8_grad/tuple/control_dependencyIdentity2gradients_4/add_20_grad/tuple/control_dependency_1(^gradients_4/add_8_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
á
1gradients_4/add_8_grad/tuple/control_dependency_1Identity2gradients_4/add_20_grad/tuple/control_dependency_1(^gradients_4/add_8_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 

gradients_4/Square_grad/ConstConst^gradients_4/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
z
gradients_4/Square_grad/MulMulsub_2gradients_4/Square_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_4/Square_grad/Mul_1Mulgradients_4/Mean_grad/truedivgradients_4/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
$gradients_4/Sum_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ˇ
gradients_4/Sum_4_grad/ReshapeReshape/gradients_4/add_8_grad/tuple/control_dependency$gradients_4/Sum_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
m
gradients_4/Sum_4_grad/ConstConst*
valueB"2      *
dtype0*
_output_shapes
:

gradients_4/Sum_4_grad/TileTilegradients_4/Sum_4_grad/Reshapegradients_4/Sum_4_grad/Const*

Tmultiples0*
T0*
_output_shapes
:	2
u
$gradients_4/Sum_5_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
š
gradients_4/Sum_5_grad/ReshapeReshape1gradients_4/add_8_grad/tuple/control_dependency_1$gradients_4/Sum_5_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
m
gradients_4/Sum_5_grad/ConstConst*
valueB"      *
dtype0*
_output_shapes
:

gradients_4/Sum_5_grad/TileTilegradients_4/Sum_5_grad/Reshapegradients_4/Sum_5_grad/Const*

Tmultiples0*
T0* 
_output_shapes
:

i
gradients_4/sub_2_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
m
gradients_4/sub_2_grad/Shape_1ShapeRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_4/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_2_grad/Shapegradients_4/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
gradients_4/sub_2_grad/SumSumgradients_4/Square_grad/Mul_1,gradients_4/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_4/sub_2_grad/ReshapeReshapegradients_4/sub_2_grad/Sumgradients_4/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients_4/sub_2_grad/Sum_1Sumgradients_4/Square_grad/Mul_1.gradients_4/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_4/sub_2_grad/NegNeggradients_4/sub_2_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_4/sub_2_grad/Reshape_1Reshapegradients_4/sub_2_grad/Neggradients_4/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_4/sub_2_grad/tuple/group_depsNoOp^gradients_4/sub_2_grad/Reshape!^gradients_4/sub_2_grad/Reshape_1
ę
/gradients_4/sub_2_grad/tuple/control_dependencyIdentitygradients_4/sub_2_grad/Reshape(^gradients_4/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/sub_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_4/sub_2_grad/tuple/control_dependency_1Identity gradients_4/sub_2_grad/Reshape_1(^gradients_4/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_4/sub_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_4/Abs_4_grad/SignSignRegress/w1_reg/read*
T0*
_output_shapes
:	2

gradients_4/Abs_4_grad/mulMulgradients_4/Sum_4_grad/Tilegradients_4/Abs_4_grad/Sign*
T0*
_output_shapes
:	2
c
gradients_4/Abs_5_grad/SignSignRegress/w2_reg/read*
T0* 
_output_shapes
:


gradients_4/Abs_5_grad/mulMulgradients_4/Sum_5_grad/Tilegradients_4/Abs_5_grad/Sign*
T0* 
_output_shapes
:

ą
,gradients_4/Regress/Sigmoid_grad/SigmoidGradSigmoidGradRegress/Sigmoid1gradients_4/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_4/Regress/add_3_grad/ShapeShapeRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
p
&gradients_4/Regress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_4/Regress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_4/Regress/add_3_grad/Shape&gradients_4/Regress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Í
"gradients_4/Regress/add_3_grad/SumSum,gradients_4/Regress/Sigmoid_grad/SigmoidGrad4gradients_4/Regress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ť
&gradients_4/Regress/add_3_grad/ReshapeReshape"gradients_4/Regress/add_3_grad/Sum$gradients_4/Regress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
$gradients_4/Regress/add_3_grad/Sum_1Sum,gradients_4/Regress/Sigmoid_grad/SigmoidGrad6gradients_4/Regress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
´
(gradients_4/Regress/add_3_grad/Reshape_1Reshape$gradients_4/Regress/add_3_grad/Sum_1&gradients_4/Regress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

/gradients_4/Regress/add_3_grad/tuple/group_depsNoOp'^gradients_4/Regress/add_3_grad/Reshape)^gradients_4/Regress/add_3_grad/Reshape_1

7gradients_4/Regress/add_3_grad/tuple/control_dependencyIdentity&gradients_4/Regress/add_3_grad/Reshape0^gradients_4/Regress/add_3_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Regress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_4/Regress/add_3_grad/tuple/control_dependency_1Identity(gradients_4/Regress/add_3_grad/Reshape_10^gradients_4/Regress/add_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/add_3_grad/Reshape_1*
_output_shapes
:
Ů
(gradients_4/Regress/MatMul_3_grad/MatMulMatMul7gradients_4/Regress/add_3_grad/tuple/control_dependencyRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
*gradients_4/Regress/MatMul_3_grad/MatMul_1MatMulRegress/Relu_27gradients_4/Regress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

2gradients_4/Regress/MatMul_3_grad/tuple/group_depsNoOp)^gradients_4/Regress/MatMul_3_grad/MatMul+^gradients_4/Regress/MatMul_3_grad/MatMul_1

:gradients_4/Regress/MatMul_3_grad/tuple/control_dependencyIdentity(gradients_4/Regress/MatMul_3_grad/MatMul3^gradients_4/Regress/MatMul_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_4/Regress/MatMul_3_grad/tuple/control_dependency_1Identity*gradients_4/Regress/MatMul_3_grad/MatMul_13^gradients_4/Regress/MatMul_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_4/Regress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
ł
(gradients_4/Regress/Relu_2_grad/ReluGradReluGrad:gradients_4/Regress/MatMul_3_grad/tuple/control_dependencyRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_4/Regress/add_2_grad/ShapeShapeRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
q
&gradients_4/Regress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_4/Regress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_4/Regress/add_2_grad/Shape&gradients_4/Regress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_4/Regress/add_2_grad/SumSum(gradients_4/Regress/Relu_2_grad/ReluGrad4gradients_4/Regress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_4/Regress/add_2_grad/ReshapeReshape"gradients_4/Regress/add_2_grad/Sum$gradients_4/Regress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_4/Regress/add_2_grad/Sum_1Sum(gradients_4/Regress/Relu_2_grad/ReluGrad6gradients_4/Regress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_4/Regress/add_2_grad/Reshape_1Reshape$gradients_4/Regress/add_2_grad/Sum_1&gradients_4/Regress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_4/Regress/add_2_grad/tuple/group_depsNoOp'^gradients_4/Regress/add_2_grad/Reshape)^gradients_4/Regress/add_2_grad/Reshape_1

7gradients_4/Regress/add_2_grad/tuple/control_dependencyIdentity&gradients_4/Regress/add_2_grad/Reshape0^gradients_4/Regress/add_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Regress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_4/Regress/add_2_grad/tuple/control_dependency_1Identity(gradients_4/Regress/add_2_grad/Reshape_10^gradients_4/Regress/add_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/add_2_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_4/Regress/MatMul_2_grad/MatMulMatMul7gradients_4/Regress/add_2_grad/tuple/control_dependencyRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
*gradients_4/Regress/MatMul_2_grad/MatMul_1MatMulRegress/Relu_17gradients_4/Regress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_4/Regress/MatMul_2_grad/tuple/group_depsNoOp)^gradients_4/Regress/MatMul_2_grad/MatMul+^gradients_4/Regress/MatMul_2_grad/MatMul_1

:gradients_4/Regress/MatMul_2_grad/tuple/control_dependencyIdentity(gradients_4/Regress/MatMul_2_grad/MatMul3^gradients_4/Regress/MatMul_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_4/Regress/MatMul_2_grad/tuple/control_dependency_1Identity*gradients_4/Regress/MatMul_2_grad/MatMul_13^gradients_4/Regress/MatMul_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_4/Regress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

ł
(gradients_4/Regress/Relu_1_grad/ReluGradReluGrad:gradients_4/Regress/MatMul_2_grad/tuple/control_dependencyRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_4/Regress/add_1_grad/ShapeShapeRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
q
&gradients_4/Regress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_4/Regress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_4/Regress/add_1_grad/Shape&gradients_4/Regress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_4/Regress/add_1_grad/SumSum(gradients_4/Regress/Relu_1_grad/ReluGrad4gradients_4/Regress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_4/Regress/add_1_grad/ReshapeReshape"gradients_4/Regress/add_1_grad/Sum$gradients_4/Regress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_4/Regress/add_1_grad/Sum_1Sum(gradients_4/Regress/Relu_1_grad/ReluGrad6gradients_4/Regress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_4/Regress/add_1_grad/Reshape_1Reshape$gradients_4/Regress/add_1_grad/Sum_1&gradients_4/Regress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_4/Regress/add_1_grad/tuple/group_depsNoOp'^gradients_4/Regress/add_1_grad/Reshape)^gradients_4/Regress/add_1_grad/Reshape_1

7gradients_4/Regress/add_1_grad/tuple/control_dependencyIdentity&gradients_4/Regress/add_1_grad/Reshape0^gradients_4/Regress/add_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Regress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_4/Regress/add_1_grad/tuple/control_dependency_1Identity(gradients_4/Regress/add_1_grad/Reshape_10^gradients_4/Regress/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/add_1_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_4/Regress/MatMul_1_grad/MatMulMatMul7gradients_4/Regress/add_1_grad/tuple/control_dependencyRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
*gradients_4/Regress/MatMul_1_grad/MatMul_1MatMulRegress/Relu7gradients_4/Regress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_4/Regress/MatMul_1_grad/tuple/group_depsNoOp)^gradients_4/Regress/MatMul_1_grad/MatMul+^gradients_4/Regress/MatMul_1_grad/MatMul_1

:gradients_4/Regress/MatMul_1_grad/tuple/control_dependencyIdentity(gradients_4/Regress/MatMul_1_grad/MatMul3^gradients_4/Regress/MatMul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_4/Regress/MatMul_1_grad/tuple/control_dependency_1Identity*gradients_4/Regress/MatMul_1_grad/MatMul_13^gradients_4/Regress/MatMul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_4/Regress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ż
&gradients_4/Regress/Relu_grad/ReluGradReluGrad:gradients_4/Regress/MatMul_1_grad/tuple/control_dependencyRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
gradients_4/AddNAddNgradients_4/Abs_5_grad/mul<gradients_4/Regress/MatMul_1_grad/tuple/control_dependency_1*
T0*-
_class#
!loc:@gradients_4/Abs_5_grad/mul*
N* 
_output_shapes
:

p
"gradients_4/Regress/add_grad/ShapeShapeRegress/MatMul*
T0*
out_type0*
_output_shapes
:
o
$gradients_4/Regress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ň
2gradients_4/Regress/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_4/Regress/add_grad/Shape$gradients_4/Regress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ă
 gradients_4/Regress/add_grad/SumSum&gradients_4/Regress/Relu_grad/ReluGrad2gradients_4/Regress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ś
$gradients_4/Regress/add_grad/ReshapeReshape gradients_4/Regress/add_grad/Sum"gradients_4/Regress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
"gradients_4/Regress/add_grad/Sum_1Sum&gradients_4/Regress/Relu_grad/ReluGrad4gradients_4/Regress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ż
&gradients_4/Regress/add_grad/Reshape_1Reshape"gradients_4/Regress/add_grad/Sum_1$gradients_4/Regress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

-gradients_4/Regress/add_grad/tuple/group_depsNoOp%^gradients_4/Regress/add_grad/Reshape'^gradients_4/Regress/add_grad/Reshape_1

5gradients_4/Regress/add_grad/tuple/control_dependencyIdentity$gradients_4/Regress/add_grad/Reshape.^gradients_4/Regress/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_4/Regress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
7gradients_4/Regress/add_grad/tuple/control_dependency_1Identity&gradients_4/Regress/add_grad/Reshape_1.^gradients_4/Regress/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Regress/add_grad/Reshape_1*
_output_shapes	
:
Ô
&gradients_4/Regress/MatMul_grad/MatMulMatMul5gradients_4/Regress/add_grad/tuple/control_dependencyRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Č
(gradients_4/Regress/MatMul_grad/MatMul_1MatMulPlaceholder_15gradients_4/Regress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	2

0gradients_4/Regress/MatMul_grad/tuple/group_depsNoOp'^gradients_4/Regress/MatMul_grad/MatMul)^gradients_4/Regress/MatMul_grad/MatMul_1

8gradients_4/Regress/MatMul_grad/tuple/control_dependencyIdentity&gradients_4/Regress/MatMul_grad/MatMul1^gradients_4/Regress/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_4/Regress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

:gradients_4/Regress/MatMul_grad/tuple/control_dependency_1Identity(gradients_4/Regress/MatMul_grad/MatMul_11^gradients_4/Regress/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_4/Regress/MatMul_grad/MatMul_1*
_output_shapes
:	2
Ô
gradients_4/AddN_1AddNgradients_4/Abs_4_grad/mul:gradients_4/Regress/MatMul_grad/tuple/control_dependency_1*
T0*-
_class#
!loc:@gradients_4/Abs_4_grad/mul*
N*
_output_shapes
:	2

beta1_power_2/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta1_power_2/readIdentitybeta1_power_2*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 

beta2_power_2/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta2_power_2/readIdentitybeta2_power_2*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Ť
7Regress/w1_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

-Regress/w1_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
ö
'Regress/w1_reg/Adam_2/Initializer/zerosFill7Regress/w1_reg/Adam_2/Initializer/zeros/shape_as_tensor-Regress/w1_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ž
Regress/w1_reg/Adam_2
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ü
Regress/w1_reg/Adam_2/AssignAssignRegress/w1_reg/Adam_2'Regress/w1_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam_2/readIdentityRegress/w1_reg/Adam_2*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ť
7Regress/w1_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

-Regress/w1_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
ö
'Regress/w1_reg/Adam_3/Initializer/zerosFill7Regress/w1_reg/Adam_3/Initializer/zeros/shape_as_tensor-Regress/w1_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ž
Regress/w1_reg/Adam_3
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ü
Regress/w1_reg/Adam_3/AssignAssignRegress/w1_reg/Adam_3'Regress/w1_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam_3/readIdentityRegress/w1_reg/Adam_3*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2

*Regress/bias1_reg/Adam_2/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias1_reg/Adam_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias1_reg/Adam_2/AssignAssignRegress/bias1_reg/Adam_2*Regress/bias1_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam_2/readIdentityRegress/bias1_reg/Adam_2*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:

*Regress/bias1_reg/Adam_3/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias1_reg/Adam_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias1_reg/Adam_3/AssignAssignRegress/bias1_reg/Adam_3*Regress/bias1_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam_3/readIdentityRegress/bias1_reg/Adam_3*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:
Ť
7Regress/w2_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

-Regress/w2_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w2_reg/Adam_2/Initializer/zerosFill7Regress/w2_reg/Adam_2/Initializer/zeros/shape_as_tensor-Regress/w2_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

°
Regress/w2_reg/Adam_2
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w2_reg/Adam_2/AssignAssignRegress/w2_reg/Adam_2'Regress/w2_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam_2/readIdentityRegress/w2_reg/Adam_2*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Ť
7Regress/w2_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

-Regress/w2_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w2_reg/Adam_3/Initializer/zerosFill7Regress/w2_reg/Adam_3/Initializer/zeros/shape_as_tensor-Regress/w2_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

°
Regress/w2_reg/Adam_3
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w2_reg/Adam_3/AssignAssignRegress/w2_reg/Adam_3'Regress/w2_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam_3/readIdentityRegress/w2_reg/Adam_3*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:


*Regress/bias2_reg/Adam_2/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias2_reg/Adam_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias2_reg/Adam_2/AssignAssignRegress/bias2_reg/Adam_2*Regress/bias2_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam_2/readIdentityRegress/bias2_reg/Adam_2*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:

*Regress/bias2_reg/Adam_3/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias2_reg/Adam_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias2_reg/Adam_3/AssignAssignRegress/bias2_reg/Adam_3*Regress/bias2_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam_3/readIdentityRegress/bias2_reg/Adam_3*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:
Ť
7Regress/w3_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

-Regress/w3_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w3_reg/Adam_2/Initializer/zerosFill7Regress/w3_reg/Adam_2/Initializer/zeros/shape_as_tensor-Regress/w3_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

°
Regress/w3_reg/Adam_2
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w3_reg/Adam_2/AssignAssignRegress/w3_reg/Adam_2'Regress/w3_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam_2/readIdentityRegress/w3_reg/Adam_2*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Ť
7Regress/w3_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

-Regress/w3_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w3_reg/Adam_3/Initializer/zerosFill7Regress/w3_reg/Adam_3/Initializer/zeros/shape_as_tensor-Regress/w3_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

°
Regress/w3_reg/Adam_3
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w3_reg/Adam_3/AssignAssignRegress/w3_reg/Adam_3'Regress/w3_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam_3/readIdentityRegress/w3_reg/Adam_3*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:


*Regress/bias3_reg/Adam_2/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias3_reg/Adam_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias3_reg/Adam_2/AssignAssignRegress/bias3_reg/Adam_2*Regress/bias3_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam_2/readIdentityRegress/bias3_reg/Adam_2*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:

*Regress/bias3_reg/Adam_3/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias3_reg/Adam_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias3_reg/Adam_3/AssignAssignRegress/bias3_reg/Adam_3*Regress/bias3_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam_3/readIdentityRegress/bias3_reg/Adam_3*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:
Ą
'Regress/w4_reg/Adam_2/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ž
Regress/w4_reg/Adam_2
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
Regress/w4_reg/Adam_2/AssignAssignRegress/w4_reg/Adam_2'Regress/w4_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam_2/readIdentityRegress/w4_reg/Adam_2*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	
Ą
'Regress/w4_reg/Adam_3/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ž
Regress/w4_reg/Adam_3
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
Regress/w4_reg/Adam_3/AssignAssignRegress/w4_reg/Adam_3'Regress/w4_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam_3/readIdentityRegress/w4_reg/Adam_3*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	

*Regress/bias4_reg/Adam_2/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
Ş
Regress/bias4_reg/Adam_2
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ă
Regress/bias4_reg/Adam_2/AssignAssignRegress/bias4_reg/Adam_2*Regress/bias4_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam_2/readIdentityRegress/bias4_reg/Adam_2*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:

*Regress/bias4_reg/Adam_3/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
Ş
Regress/bias4_reg/Adam_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ă
Regress/bias4_reg/Adam_3/AssignAssignRegress/bias4_reg/Adam_3*Regress/bias4_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam_3/readIdentityRegress/bias4_reg/Adam_3*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:
Y
Adam_2/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_2/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_2/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_2/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ă
&Adam_2/update_Regress/w1_reg/ApplyAdam	ApplyAdamRegress/w1_regRegress/w1_reg/Adam_2Regress/w1_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilongradients_4/AddN_1*
use_locking( *
T0*!
_class
loc:@Regress/w1_reg*
use_nesterov( *
_output_shapes
:	2

)Adam_2/update_Regress/bias1_reg/ApplyAdam	ApplyAdamRegress/bias1_regRegress/bias1_reg/Adam_2Regress/bias1_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon7gradients_4/Regress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
use_nesterov( *
_output_shapes	
:
â
&Adam_2/update_Regress/w2_reg/ApplyAdam	ApplyAdamRegress/w2_regRegress/w2_reg/Adam_2Regress/w2_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilongradients_4/AddN*
use_locking( *
T0*!
_class
loc:@Regress/w2_reg*
use_nesterov( * 
_output_shapes
:


)Adam_2/update_Regress/bias2_reg/ApplyAdam	ApplyAdamRegress/bias2_regRegress/bias2_reg/Adam_2Regress/bias2_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon9gradients_4/Regress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias2_reg*
use_nesterov( *
_output_shapes	
:

&Adam_2/update_Regress/w3_reg/ApplyAdam	ApplyAdamRegress/w3_regRegress/w3_reg/Adam_2Regress/w3_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon<gradients_4/Regress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w3_reg*
use_nesterov( * 
_output_shapes
:


)Adam_2/update_Regress/bias3_reg/ApplyAdam	ApplyAdamRegress/bias3_regRegress/bias3_reg/Adam_2Regress/bias3_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon9gradients_4/Regress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias3_reg*
use_nesterov( *
_output_shapes	
:

&Adam_2/update_Regress/w4_reg/ApplyAdam	ApplyAdamRegress/w4_regRegress/w4_reg/Adam_2Regress/w4_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon<gradients_4/Regress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w4_reg*
use_nesterov( *
_output_shapes
:	

)Adam_2/update_Regress/bias4_reg/ApplyAdam	ApplyAdamRegress/bias4_regRegress/bias4_reg/Adam_2Regress/bias4_reg/Adam_3beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon9gradients_4/Regress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias4_reg*
use_nesterov( *
_output_shapes
:
Î

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1*^Adam_2/update_Regress/bias1_reg/ApplyAdam*^Adam_2/update_Regress/bias2_reg/ApplyAdam*^Adam_2/update_Regress/bias3_reg/ApplyAdam*^Adam_2/update_Regress/bias4_reg/ApplyAdam'^Adam_2/update_Regress/w1_reg/ApplyAdam'^Adam_2/update_Regress/w2_reg/ApplyAdam'^Adam_2/update_Regress/w3_reg/ApplyAdam'^Adam_2/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
˘
Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Đ
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2*^Adam_2/update_Regress/bias1_reg/ApplyAdam*^Adam_2/update_Regress/bias2_reg/ApplyAdam*^Adam_2/update_Regress/bias3_reg/ApplyAdam*^Adam_2/update_Regress/bias4_reg/ApplyAdam'^Adam_2/update_Regress/w1_reg/ApplyAdam'^Adam_2/update_Regress/w2_reg/ApplyAdam'^Adam_2/update_Regress/w3_reg/ApplyAdam'^Adam_2/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Ś
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_1*^Adam_2/update_Regress/bias1_reg/ApplyAdam*^Adam_2/update_Regress/bias2_reg/ApplyAdam*^Adam_2/update_Regress/bias3_reg/ApplyAdam*^Adam_2/update_Regress/bias4_reg/ApplyAdam'^Adam_2/update_Regress/w1_reg/ApplyAdam'^Adam_2/update_Regress/w2_reg/ApplyAdam'^Adam_2/update_Regress/w3_reg/ApplyAdam'^Adam_2/update_Regress/w4_reg/ApplyAdam
T
gradients_5/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_5/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_5/FillFillgradients_5/Shapegradients_5/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
C
(gradients_5/add_22_grad/tuple/group_depsNoOp^gradients_5/Fill
ż
0gradients_5/add_22_grad/tuple/control_dependencyIdentitygradients_5/Fill)^gradients_5/add_22_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_5/Fill*
_output_shapes
: 
Á
2gradients_5/add_22_grad/tuple/control_dependency_1Identitygradients_5/Fill)^gradients_5/add_22_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_5/Fill*
_output_shapes
: 
t
#gradients_5/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ś
gradients_5/Mean_grad/ReshapeReshape0gradients_5/add_22_grad/tuple/control_dependency#gradients_5/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
a
gradients_5/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
˘
gradients_5/Mean_grad/TileTilegradients_5/Mean_grad/Reshapegradients_5/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_5/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
`
gradients_5/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients_5/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients_5/Mean_grad/ProdProdgradients_5/Mean_grad/Shape_1gradients_5/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients_5/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients_5/Mean_grad/Prod_1Prodgradients_5/Mean_grad/Shape_2gradients_5/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
a
gradients_5/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_5/Mean_grad/MaximumMaximumgradients_5/Mean_grad/Prod_1gradients_5/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients_5/Mean_grad/floordivFloorDivgradients_5/Mean_grad/Prodgradients_5/Mean_grad/Maximum*
T0*
_output_shapes
: 

gradients_5/Mean_grad/CastCastgradients_5/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_5/Mean_grad/truedivRealDivgradients_5/Mean_grad/Tilegradients_5/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
(gradients_5/add_12_grad/tuple/group_depsNoOp3^gradients_5/add_22_grad/tuple/control_dependency_1
á
0gradients_5/add_12_grad/tuple/control_dependencyIdentity2gradients_5/add_22_grad/tuple/control_dependency_1)^gradients_5/add_12_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_5/Fill*
_output_shapes
: 
ă
2gradients_5/add_12_grad/tuple/control_dependency_1Identity2gradients_5/add_22_grad/tuple/control_dependency_1)^gradients_5/add_12_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_5/Fill*
_output_shapes
: 

gradients_5/Square_grad/ConstConst^gradients_5/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
z
gradients_5/Square_grad/MulMulsub_2gradients_5/Square_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_5/Square_grad/Mul_1Mulgradients_5/Mean_grad/truedivgradients_5/Square_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_5/L2Loss_4_grad/mulMulRegress/w1_reg/read0gradients_5/add_12_grad/tuple/control_dependency*
T0*
_output_shapes
:	2

gradients_5/L2Loss_5_grad/mulMulRegress/w2_reg/read2gradients_5/add_12_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

i
gradients_5/sub_2_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
m
gradients_5/sub_2_grad/Shape_1ShapeRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_5/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/sub_2_grad/Shapegradients_5/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
gradients_5/sub_2_grad/SumSumgradients_5/Square_grad/Mul_1,gradients_5/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_5/sub_2_grad/ReshapeReshapegradients_5/sub_2_grad/Sumgradients_5/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
gradients_5/sub_2_grad/Sum_1Sumgradients_5/Square_grad/Mul_1.gradients_5/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_5/sub_2_grad/NegNeggradients_5/sub_2_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_5/sub_2_grad/Reshape_1Reshapegradients_5/sub_2_grad/Neggradients_5/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_5/sub_2_grad/tuple/group_depsNoOp^gradients_5/sub_2_grad/Reshape!^gradients_5/sub_2_grad/Reshape_1
ę
/gradients_5/sub_2_grad/tuple/control_dependencyIdentitygradients_5/sub_2_grad/Reshape(^gradients_5/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_5/sub_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_5/sub_2_grad/tuple/control_dependency_1Identity gradients_5/sub_2_grad/Reshape_1(^gradients_5/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_5/sub_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
,gradients_5/Regress/Sigmoid_grad/SigmoidGradSigmoidGradRegress/Sigmoid1gradients_5/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_5/Regress/add_3_grad/ShapeShapeRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
p
&gradients_5/Regress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_5/Regress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_5/Regress/add_3_grad/Shape&gradients_5/Regress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Í
"gradients_5/Regress/add_3_grad/SumSum,gradients_5/Regress/Sigmoid_grad/SigmoidGrad4gradients_5/Regress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ť
&gradients_5/Regress/add_3_grad/ReshapeReshape"gradients_5/Regress/add_3_grad/Sum$gradients_5/Regress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
$gradients_5/Regress/add_3_grad/Sum_1Sum,gradients_5/Regress/Sigmoid_grad/SigmoidGrad6gradients_5/Regress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
´
(gradients_5/Regress/add_3_grad/Reshape_1Reshape$gradients_5/Regress/add_3_grad/Sum_1&gradients_5/Regress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

/gradients_5/Regress/add_3_grad/tuple/group_depsNoOp'^gradients_5/Regress/add_3_grad/Reshape)^gradients_5/Regress/add_3_grad/Reshape_1

7gradients_5/Regress/add_3_grad/tuple/control_dependencyIdentity&gradients_5/Regress/add_3_grad/Reshape0^gradients_5/Regress/add_3_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_5/Regress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_5/Regress/add_3_grad/tuple/control_dependency_1Identity(gradients_5/Regress/add_3_grad/Reshape_10^gradients_5/Regress/add_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/add_3_grad/Reshape_1*
_output_shapes
:
Ů
(gradients_5/Regress/MatMul_3_grad/MatMulMatMul7gradients_5/Regress/add_3_grad/tuple/control_dependencyRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
*gradients_5/Regress/MatMul_3_grad/MatMul_1MatMulRegress/Relu_27gradients_5/Regress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	

2gradients_5/Regress/MatMul_3_grad/tuple/group_depsNoOp)^gradients_5/Regress/MatMul_3_grad/MatMul+^gradients_5/Regress/MatMul_3_grad/MatMul_1

:gradients_5/Regress/MatMul_3_grad/tuple/control_dependencyIdentity(gradients_5/Regress/MatMul_3_grad/MatMul3^gradients_5/Regress/MatMul_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_5/Regress/MatMul_3_grad/tuple/control_dependency_1Identity*gradients_5/Regress/MatMul_3_grad/MatMul_13^gradients_5/Regress/MatMul_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_5/Regress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
ł
(gradients_5/Regress/Relu_2_grad/ReluGradReluGrad:gradients_5/Regress/MatMul_3_grad/tuple/control_dependencyRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_5/Regress/add_2_grad/ShapeShapeRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
q
&gradients_5/Regress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_5/Regress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_5/Regress/add_2_grad/Shape&gradients_5/Regress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_5/Regress/add_2_grad/SumSum(gradients_5/Regress/Relu_2_grad/ReluGrad4gradients_5/Regress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_5/Regress/add_2_grad/ReshapeReshape"gradients_5/Regress/add_2_grad/Sum$gradients_5/Regress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_5/Regress/add_2_grad/Sum_1Sum(gradients_5/Regress/Relu_2_grad/ReluGrad6gradients_5/Regress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_5/Regress/add_2_grad/Reshape_1Reshape$gradients_5/Regress/add_2_grad/Sum_1&gradients_5/Regress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_5/Regress/add_2_grad/tuple/group_depsNoOp'^gradients_5/Regress/add_2_grad/Reshape)^gradients_5/Regress/add_2_grad/Reshape_1

7gradients_5/Regress/add_2_grad/tuple/control_dependencyIdentity&gradients_5/Regress/add_2_grad/Reshape0^gradients_5/Regress/add_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_5/Regress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_5/Regress/add_2_grad/tuple/control_dependency_1Identity(gradients_5/Regress/add_2_grad/Reshape_10^gradients_5/Regress/add_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/add_2_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_5/Regress/MatMul_2_grad/MatMulMatMul7gradients_5/Regress/add_2_grad/tuple/control_dependencyRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
*gradients_5/Regress/MatMul_2_grad/MatMul_1MatMulRegress/Relu_17gradients_5/Regress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_5/Regress/MatMul_2_grad/tuple/group_depsNoOp)^gradients_5/Regress/MatMul_2_grad/MatMul+^gradients_5/Regress/MatMul_2_grad/MatMul_1

:gradients_5/Regress/MatMul_2_grad/tuple/control_dependencyIdentity(gradients_5/Regress/MatMul_2_grad/MatMul3^gradients_5/Regress/MatMul_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_5/Regress/MatMul_2_grad/tuple/control_dependency_1Identity*gradients_5/Regress/MatMul_2_grad/MatMul_13^gradients_5/Regress/MatMul_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_5/Regress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

ł
(gradients_5/Regress/Relu_1_grad/ReluGradReluGrad:gradients_5/Regress/MatMul_2_grad/tuple/control_dependencyRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
$gradients_5/Regress/add_1_grad/ShapeShapeRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
q
&gradients_5/Regress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ř
4gradients_5/Regress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_5/Regress/add_1_grad/Shape&gradients_5/Regress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
"gradients_5/Regress/add_1_grad/SumSum(gradients_5/Regress/Relu_1_grad/ReluGrad4gradients_5/Regress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ź
&gradients_5/Regress/add_1_grad/ReshapeReshape"gradients_5/Regress/add_1_grad/Sum$gradients_5/Regress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
$gradients_5/Regress/add_1_grad/Sum_1Sum(gradients_5/Regress/Relu_1_grad/ReluGrad6gradients_5/Regress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ľ
(gradients_5/Regress/add_1_grad/Reshape_1Reshape$gradients_5/Regress/add_1_grad/Sum_1&gradients_5/Regress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

/gradients_5/Regress/add_1_grad/tuple/group_depsNoOp'^gradients_5/Regress/add_1_grad/Reshape)^gradients_5/Regress/add_1_grad/Reshape_1

7gradients_5/Regress/add_1_grad/tuple/control_dependencyIdentity&gradients_5/Regress/add_1_grad/Reshape0^gradients_5/Regress/add_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_5/Regress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_5/Regress/add_1_grad/tuple/control_dependency_1Identity(gradients_5/Regress/add_1_grad/Reshape_10^gradients_5/Regress/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/add_1_grad/Reshape_1*
_output_shapes	
:
Ů
(gradients_5/Regress/MatMul_1_grad/MatMulMatMul7gradients_5/Regress/add_1_grad/tuple/control_dependencyRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
*gradients_5/Regress/MatMul_1_grad/MatMul_1MatMulRegress/Relu7gradients_5/Regress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


2gradients_5/Regress/MatMul_1_grad/tuple/group_depsNoOp)^gradients_5/Regress/MatMul_1_grad/MatMul+^gradients_5/Regress/MatMul_1_grad/MatMul_1

:gradients_5/Regress/MatMul_1_grad/tuple/control_dependencyIdentity(gradients_5/Regress/MatMul_1_grad/MatMul3^gradients_5/Regress/MatMul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_5/Regress/MatMul_1_grad/tuple/control_dependency_1Identity*gradients_5/Regress/MatMul_1_grad/MatMul_13^gradients_5/Regress/MatMul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_5/Regress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ż
&gradients_5/Regress/Relu_grad/ReluGradReluGrad:gradients_5/Regress/MatMul_1_grad/tuple/control_dependencyRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
gradients_5/AddNAddNgradients_5/L2Loss_5_grad/mul<gradients_5/Regress/MatMul_1_grad/tuple/control_dependency_1*
T0*0
_class&
$"loc:@gradients_5/L2Loss_5_grad/mul*
N* 
_output_shapes
:

p
"gradients_5/Regress/add_grad/ShapeShapeRegress/MatMul*
T0*
out_type0*
_output_shapes
:
o
$gradients_5/Regress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ň
2gradients_5/Regress/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_5/Regress/add_grad/Shape$gradients_5/Regress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ă
 gradients_5/Regress/add_grad/SumSum&gradients_5/Regress/Relu_grad/ReluGrad2gradients_5/Regress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ś
$gradients_5/Regress/add_grad/ReshapeReshape gradients_5/Regress/add_grad/Sum"gradients_5/Regress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
"gradients_5/Regress/add_grad/Sum_1Sum&gradients_5/Regress/Relu_grad/ReluGrad4gradients_5/Regress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ż
&gradients_5/Regress/add_grad/Reshape_1Reshape"gradients_5/Regress/add_grad/Sum_1$gradients_5/Regress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

-gradients_5/Regress/add_grad/tuple/group_depsNoOp%^gradients_5/Regress/add_grad/Reshape'^gradients_5/Regress/add_grad/Reshape_1

5gradients_5/Regress/add_grad/tuple/control_dependencyIdentity$gradients_5/Regress/add_grad/Reshape.^gradients_5/Regress/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_5/Regress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
7gradients_5/Regress/add_grad/tuple/control_dependency_1Identity&gradients_5/Regress/add_grad/Reshape_1.^gradients_5/Regress/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_5/Regress/add_grad/Reshape_1*
_output_shapes	
:
Ô
&gradients_5/Regress/MatMul_grad/MatMulMatMul5gradients_5/Regress/add_grad/tuple/control_dependencyRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Č
(gradients_5/Regress/MatMul_grad/MatMul_1MatMulPlaceholder_15gradients_5/Regress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	2

0gradients_5/Regress/MatMul_grad/tuple/group_depsNoOp'^gradients_5/Regress/MatMul_grad/MatMul)^gradients_5/Regress/MatMul_grad/MatMul_1

8gradients_5/Regress/MatMul_grad/tuple/control_dependencyIdentity&gradients_5/Regress/MatMul_grad/MatMul1^gradients_5/Regress/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_5/Regress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

:gradients_5/Regress/MatMul_grad/tuple/control_dependency_1Identity(gradients_5/Regress/MatMul_grad/MatMul_11^gradients_5/Regress/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_5/Regress/MatMul_grad/MatMul_1*
_output_shapes
:	2
Ú
gradients_5/AddN_1AddNgradients_5/L2Loss_4_grad/mul:gradients_5/Regress/MatMul_grad/tuple/control_dependency_1*
T0*0
_class&
$"loc:@gradients_5/L2Loss_4_grad/mul*
N*
_output_shapes
:	2

beta1_power_3/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta1_power_3/AssignAssignbeta1_power_3beta1_power_3/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta1_power_3/readIdentitybeta1_power_3*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 

beta2_power_3/initial_valueConst*$
_class
loc:@Regress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_3
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
ş
beta2_power_3/AssignAssignbeta2_power_3beta2_power_3/initial_value*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
t
beta2_power_3/readIdentitybeta2_power_3*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Ť
7Regress/w1_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

-Regress/w1_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
ö
'Regress/w1_reg/Adam_4/Initializer/zerosFill7Regress/w1_reg/Adam_4/Initializer/zeros/shape_as_tensor-Regress/w1_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ž
Regress/w1_reg/Adam_4
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ü
Regress/w1_reg/Adam_4/AssignAssignRegress/w1_reg/Adam_4'Regress/w1_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam_4/readIdentityRegress/w1_reg/Adam_4*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ť
7Regress/w1_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"2      *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
:

-Regress/w1_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w1_reg*
dtype0*
_output_shapes
: 
ö
'Regress/w1_reg/Adam_5/Initializer/zerosFill7Regress/w1_reg/Adam_5/Initializer/zeros/shape_as_tensor-Regress/w1_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2
Ž
Regress/w1_reg/Adam_5
VariableV2*
shared_name *!
_class
loc:@Regress/w1_reg*
	container *
shape:	2*
dtype0*
_output_shapes
:	2
Ü
Regress/w1_reg/Adam_5/AssignAssignRegress/w1_reg/Adam_5'Regress/w1_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2

Regress/w1_reg/Adam_5/readIdentityRegress/w1_reg/Adam_5*
T0*!
_class
loc:@Regress/w1_reg*
_output_shapes
:	2

*Regress/bias1_reg/Adam_4/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias1_reg/Adam_4
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias1_reg/Adam_4/AssignAssignRegress/bias1_reg/Adam_4*Regress/bias1_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam_4/readIdentityRegress/bias1_reg/Adam_4*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:

*Regress/bias1_reg/Adam_5/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias1_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias1_reg/Adam_5
VariableV2*
shared_name *$
_class
loc:@Regress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias1_reg/Adam_5/AssignAssignRegress/bias1_reg/Adam_5*Regress/bias1_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:

Regress/bias1_reg/Adam_5/readIdentityRegress/bias1_reg/Adam_5*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes	
:
Ť
7Regress/w2_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

-Regress/w2_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w2_reg/Adam_4/Initializer/zerosFill7Regress/w2_reg/Adam_4/Initializer/zeros/shape_as_tensor-Regress/w2_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

°
Regress/w2_reg/Adam_4
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w2_reg/Adam_4/AssignAssignRegress/w2_reg/Adam_4'Regress/w2_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam_4/readIdentityRegress/w2_reg/Adam_4*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

Ť
7Regress/w2_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
:

-Regress/w2_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w2_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w2_reg/Adam_5/Initializer/zerosFill7Regress/w2_reg/Adam_5/Initializer/zeros/shape_as_tensor-Regress/w2_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:

°
Regress/w2_reg/Adam_5
VariableV2*
shared_name *!
_class
loc:@Regress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w2_reg/Adam_5/AssignAssignRegress/w2_reg/Adam_5'Regress/w2_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:


Regress/w2_reg/Adam_5/readIdentityRegress/w2_reg/Adam_5*
T0*!
_class
loc:@Regress/w2_reg* 
_output_shapes
:


*Regress/bias2_reg/Adam_4/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias2_reg/Adam_4
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias2_reg/Adam_4/AssignAssignRegress/bias2_reg/Adam_4*Regress/bias2_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam_4/readIdentityRegress/bias2_reg/Adam_4*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:

*Regress/bias2_reg/Adam_5/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias2_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias2_reg/Adam_5
VariableV2*
shared_name *$
_class
loc:@Regress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias2_reg/Adam_5/AssignAssignRegress/bias2_reg/Adam_5*Regress/bias2_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:

Regress/bias2_reg/Adam_5/readIdentityRegress/bias2_reg/Adam_5*
T0*$
_class
loc:@Regress/bias2_reg*
_output_shapes	
:
Ť
7Regress/w3_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

-Regress/w3_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w3_reg/Adam_4/Initializer/zerosFill7Regress/w3_reg/Adam_4/Initializer/zeros/shape_as_tensor-Regress/w3_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

°
Regress/w3_reg/Adam_4
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w3_reg/Adam_4/AssignAssignRegress/w3_reg/Adam_4'Regress/w3_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam_4/readIdentityRegress/w3_reg/Adam_4*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

Ť
7Regress/w3_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
:

-Regress/w3_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Regress/w3_reg*
dtype0*
_output_shapes
: 
÷
'Regress/w3_reg/Adam_5/Initializer/zerosFill7Regress/w3_reg/Adam_5/Initializer/zeros/shape_as_tensor-Regress/w3_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:

°
Regress/w3_reg/Adam_5
VariableV2*
shared_name *!
_class
loc:@Regress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ý
Regress/w3_reg/Adam_5/AssignAssignRegress/w3_reg/Adam_5'Regress/w3_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:


Regress/w3_reg/Adam_5/readIdentityRegress/w3_reg/Adam_5*
T0*!
_class
loc:@Regress/w3_reg* 
_output_shapes
:


*Regress/bias3_reg/Adam_4/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias3_reg/Adam_4
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias3_reg/Adam_4/AssignAssignRegress/bias3_reg/Adam_4*Regress/bias3_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam_4/readIdentityRegress/bias3_reg/Adam_4*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:

*Regress/bias3_reg/Adam_5/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias3_reg*
dtype0*
_output_shapes	
:
Ź
Regress/bias3_reg/Adam_5
VariableV2*
shared_name *$
_class
loc:@Regress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ä
Regress/bias3_reg/Adam_5/AssignAssignRegress/bias3_reg/Adam_5*Regress/bias3_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:

Regress/bias3_reg/Adam_5/readIdentityRegress/bias3_reg/Adam_5*
T0*$
_class
loc:@Regress/bias3_reg*
_output_shapes	
:
Ą
'Regress/w4_reg/Adam_4/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ž
Regress/w4_reg/Adam_4
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
Regress/w4_reg/Adam_4/AssignAssignRegress/w4_reg/Adam_4'Regress/w4_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam_4/readIdentityRegress/w4_reg/Adam_4*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	
Ą
'Regress/w4_reg/Adam_5/Initializer/zerosConst*
valueB	*    *!
_class
loc:@Regress/w4_reg*
dtype0*
_output_shapes
:	
Ž
Regress/w4_reg/Adam_5
VariableV2*
shared_name *!
_class
loc:@Regress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ü
Regress/w4_reg/Adam_5/AssignAssignRegress/w4_reg/Adam_5'Regress/w4_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	

Regress/w4_reg/Adam_5/readIdentityRegress/w4_reg/Adam_5*
T0*!
_class
loc:@Regress/w4_reg*
_output_shapes
:	

*Regress/bias4_reg/Adam_4/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
Ş
Regress/bias4_reg/Adam_4
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ă
Regress/bias4_reg/Adam_4/AssignAssignRegress/bias4_reg/Adam_4*Regress/bias4_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam_4/readIdentityRegress/bias4_reg/Adam_4*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:

*Regress/bias4_reg/Adam_5/Initializer/zerosConst*
valueB*    *$
_class
loc:@Regress/bias4_reg*
dtype0*
_output_shapes
:
Ş
Regress/bias4_reg/Adam_5
VariableV2*
shared_name *$
_class
loc:@Regress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ă
Regress/bias4_reg/Adam_5/AssignAssignRegress/bias4_reg/Adam_5*Regress/bias4_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:

Regress/bias4_reg/Adam_5/readIdentityRegress/bias4_reg/Adam_5*
T0*$
_class
loc:@Regress/bias4_reg*
_output_shapes
:
Y
Adam_3/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_3/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_3/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_3/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ă
&Adam_3/update_Regress/w1_reg/ApplyAdam	ApplyAdamRegress/w1_regRegress/w1_reg/Adam_4Regress/w1_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilongradients_5/AddN_1*
use_locking( *
T0*!
_class
loc:@Regress/w1_reg*
use_nesterov( *
_output_shapes
:	2

)Adam_3/update_Regress/bias1_reg/ApplyAdam	ApplyAdamRegress/bias1_regRegress/bias1_reg/Adam_4Regress/bias1_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon7gradients_5/Regress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
use_nesterov( *
_output_shapes	
:
â
&Adam_3/update_Regress/w2_reg/ApplyAdam	ApplyAdamRegress/w2_regRegress/w2_reg/Adam_4Regress/w2_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilongradients_5/AddN*
use_locking( *
T0*!
_class
loc:@Regress/w2_reg*
use_nesterov( * 
_output_shapes
:


)Adam_3/update_Regress/bias2_reg/ApplyAdam	ApplyAdamRegress/bias2_regRegress/bias2_reg/Adam_4Regress/bias2_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon9gradients_5/Regress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias2_reg*
use_nesterov( *
_output_shapes	
:

&Adam_3/update_Regress/w3_reg/ApplyAdam	ApplyAdamRegress/w3_regRegress/w3_reg/Adam_4Regress/w3_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon<gradients_5/Regress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w3_reg*
use_nesterov( * 
_output_shapes
:


)Adam_3/update_Regress/bias3_reg/ApplyAdam	ApplyAdamRegress/bias3_regRegress/bias3_reg/Adam_4Regress/bias3_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon9gradients_5/Regress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias3_reg*
use_nesterov( *
_output_shapes	
:

&Adam_3/update_Regress/w4_reg/ApplyAdam	ApplyAdamRegress/w4_regRegress/w4_reg/Adam_4Regress/w4_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon<gradients_5/Regress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@Regress/w4_reg*
use_nesterov( *
_output_shapes
:	

)Adam_3/update_Regress/bias4_reg/ApplyAdam	ApplyAdamRegress/bias4_regRegress/bias4_reg/Adam_4Regress/bias4_reg/Adam_5beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon9gradients_5/Regress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Regress/bias4_reg*
use_nesterov( *
_output_shapes
:
Î

Adam_3/mulMulbeta1_power_3/readAdam_3/beta1*^Adam_3/update_Regress/bias1_reg/ApplyAdam*^Adam_3/update_Regress/bias2_reg/ApplyAdam*^Adam_3/update_Regress/bias3_reg/ApplyAdam*^Adam_3/update_Regress/bias4_reg/ApplyAdam'^Adam_3/update_Regress/w1_reg/ApplyAdam'^Adam_3/update_Regress/w2_reg/ApplyAdam'^Adam_3/update_Regress/w3_reg/ApplyAdam'^Adam_3/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
˘
Adam_3/AssignAssignbeta1_power_3
Adam_3/mul*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Đ
Adam_3/mul_1Mulbeta2_power_3/readAdam_3/beta2*^Adam_3/update_Regress/bias1_reg/ApplyAdam*^Adam_3/update_Regress/bias2_reg/ApplyAdam*^Adam_3/update_Regress/bias3_reg/ApplyAdam*^Adam_3/update_Regress/bias4_reg/ApplyAdam'^Adam_3/update_Regress/w1_reg/ApplyAdam'^Adam_3/update_Regress/w2_reg/ApplyAdam'^Adam_3/update_Regress/w3_reg/ApplyAdam'^Adam_3/update_Regress/w4_reg/ApplyAdam*
T0*$
_class
loc:@Regress/bias1_reg*
_output_shapes
: 
Ś
Adam_3/Assign_1Assignbeta2_power_3Adam_3/mul_1*
use_locking( *
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_3NoOp^Adam_3/Assign^Adam_3/Assign_1*^Adam_3/update_Regress/bias1_reg/ApplyAdam*^Adam_3/update_Regress/bias2_reg/ApplyAdam*^Adam_3/update_Regress/bias3_reg/ApplyAdam*^Adam_3/update_Regress/bias4_reg/ApplyAdam'^Adam_3/update_Regress/w1_reg/ApplyAdam'^Adam_3/update_Regress/w2_reg/ApplyAdam'^Adam_3/update_Regress/w3_reg/ApplyAdam'^Adam_3/update_Regress/w4_reg/ApplyAdam
T
gradients_6/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_6/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_6/FillFillgradients_6/Shapegradients_6/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
%gradients_6/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients_6/Mean_2_grad/ReshapeReshapegradients_6/Fill%gradients_6/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
e
gradients_6/Mean_2_grad/ShapeShapeSquare_2*
T0*
out_type0*
_output_shapes
:
¨
gradients_6/Mean_2_grad/TileTilegradients_6/Mean_2_grad/Reshapegradients_6/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_6/Mean_2_grad/Shape_1ShapeSquare_2*
T0*
out_type0*
_output_shapes
:
b
gradients_6/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_6/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_6/Mean_2_grad/ProdProdgradients_6/Mean_2_grad/Shape_1gradients_6/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
gradients_6/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_6/Mean_2_grad/Prod_1Prodgradients_6/Mean_2_grad/Shape_2gradients_6/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_6/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_6/Mean_2_grad/MaximumMaximumgradients_6/Mean_2_grad/Prod_1!gradients_6/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_6/Mean_2_grad/floordivFloorDivgradients_6/Mean_2_grad/Prodgradients_6/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_6/Mean_2_grad/CastCast gradients_6/Mean_2_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_6/Mean_2_grad/truedivRealDivgradients_6/Mean_2_grad/Tilegradients_6/Mean_2_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_6/Square_2_grad/ConstConst ^gradients_6/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
~
gradients_6/Square_2_grad/MulMulsub_4gradients_6/Square_2_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_6/Square_2_grad/Mul_1Mulgradients_6/Mean_2_grad/truedivgradients_6/Square_2_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients_6/sub_4_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
u
gradients_6/sub_4_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_6/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/sub_4_grad/Shapegradients_6/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
°
gradients_6/sub_4_grad/SumSumgradients_6/Square_2_grad/Mul_1,gradients_6/sub_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_6/sub_4_grad/ReshapeReshapegradients_6/sub_4_grad/Sumgradients_6/sub_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_6/sub_4_grad/Sum_1Sumgradients_6/Square_2_grad/Mul_1.gradients_6/sub_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_6/sub_4_grad/NegNeggradients_6/sub_4_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_6/sub_4_grad/Reshape_1Reshapegradients_6/sub_4_grad/Neggradients_6/sub_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_6/sub_4_grad/tuple/group_depsNoOp^gradients_6/sub_4_grad/Reshape!^gradients_6/sub_4_grad/Reshape_1
ę
/gradients_6/sub_4_grad/tuple/control_dependencyIdentitygradients_6/sub_4_grad/Reshape(^gradients_6/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_6/sub_4_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_6/sub_4_grad/tuple/control_dependency_1Identity gradients_6/sub_4_grad/Reshape_1(^gradients_6/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_6/sub_4_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
4gradients_6/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid1gradients_6/sub_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_6/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
x
.gradients_6/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_6/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_6/ResidualRegress/add_3_grad/Shape.gradients_6/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
*gradients_6/ResidualRegress/add_3_grad/SumSum4gradients_6/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients_6/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ó
.gradients_6/ResidualRegress/add_3_grad/ReshapeReshape*gradients_6/ResidualRegress/add_3_grad/Sum,gradients_6/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
,gradients_6/ResidualRegress/add_3_grad/Sum_1Sum4gradients_6/ResidualRegress/Sigmoid_grad/SigmoidGrad>gradients_6/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_6/ResidualRegress/add_3_grad/Reshape_1Reshape,gradients_6/ResidualRegress/add_3_grad/Sum_1.gradients_6/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ł
7gradients_6/ResidualRegress/add_3_grad/tuple/group_depsNoOp/^gradients_6/ResidualRegress/add_3_grad/Reshape1^gradients_6/ResidualRegress/add_3_grad/Reshape_1
Ş
?gradients_6/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity.gradients_6/ResidualRegress/add_3_grad/Reshape8^gradients_6/ResidualRegress/add_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_6/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Agradients_6/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity0gradients_6/ResidualRegress/add_3_grad/Reshape_18^gradients_6/ResidualRegress/add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ń
0gradients_6/ResidualRegress/MatMul_3_grad/MatMulMatMul?gradients_6/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
2gradients_6/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2?gradients_6/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
Ş
:gradients_6/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp1^gradients_6/ResidualRegress/MatMul_3_grad/MatMul3^gradients_6/ResidualRegress/MatMul_3_grad/MatMul_1
ľ
Bgradients_6/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity0gradients_6/ResidualRegress/MatMul_3_grad/MatMul;^gradients_6/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients_6/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity2gradients_6/ResidualRegress/MatMul_3_grad/MatMul_1;^gradients_6/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_6/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Ë
0gradients_6/ResidualRegress/Relu_2_grad/ReluGradReluGradBgradients_6/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_6/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
y
.gradients_6/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_6/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_6/ResidualRegress/add_2_grad/Shape.gradients_6/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_6/ResidualRegress/add_2_grad/SumSum0gradients_6/ResidualRegress/Relu_2_grad/ReluGrad<gradients_6/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_6/ResidualRegress/add_2_grad/ReshapeReshape*gradients_6/ResidualRegress/add_2_grad/Sum,gradients_6/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_6/ResidualRegress/add_2_grad/Sum_1Sum0gradients_6/ResidualRegress/Relu_2_grad/ReluGrad>gradients_6/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_6/ResidualRegress/add_2_grad/Reshape_1Reshape,gradients_6/ResidualRegress/add_2_grad/Sum_1.gradients_6/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_6/ResidualRegress/add_2_grad/tuple/group_depsNoOp/^gradients_6/ResidualRegress/add_2_grad/Reshape1^gradients_6/ResidualRegress/add_2_grad/Reshape_1
Ť
?gradients_6/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity.gradients_6/ResidualRegress/add_2_grad/Reshape8^gradients_6/ResidualRegress/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_6/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_6/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity0gradients_6/ResidualRegress/add_2_grad/Reshape_18^gradients_6/ResidualRegress/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_6/ResidualRegress/MatMul_2_grad/MatMulMatMul?gradients_6/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
2gradients_6/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1?gradients_6/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_6/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp1^gradients_6/ResidualRegress/MatMul_2_grad/MatMul3^gradients_6/ResidualRegress/MatMul_2_grad/MatMul_1
ľ
Bgradients_6/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_6/ResidualRegress/MatMul_2_grad/MatMul;^gradients_6/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_6/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_6/ResidualRegress/MatMul_2_grad/MatMul_1;^gradients_6/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_6/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Ë
0gradients_6/ResidualRegress/Relu_1_grad/ReluGradReluGradBgradients_6/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_6/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
y
.gradients_6/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_6/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_6/ResidualRegress/add_1_grad/Shape.gradients_6/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_6/ResidualRegress/add_1_grad/SumSum0gradients_6/ResidualRegress/Relu_1_grad/ReluGrad<gradients_6/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_6/ResidualRegress/add_1_grad/ReshapeReshape*gradients_6/ResidualRegress/add_1_grad/Sum,gradients_6/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_6/ResidualRegress/add_1_grad/Sum_1Sum0gradients_6/ResidualRegress/Relu_1_grad/ReluGrad>gradients_6/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_6/ResidualRegress/add_1_grad/Reshape_1Reshape,gradients_6/ResidualRegress/add_1_grad/Sum_1.gradients_6/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_6/ResidualRegress/add_1_grad/tuple/group_depsNoOp/^gradients_6/ResidualRegress/add_1_grad/Reshape1^gradients_6/ResidualRegress/add_1_grad/Reshape_1
Ť
?gradients_6/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity.gradients_6/ResidualRegress/add_1_grad/Reshape8^gradients_6/ResidualRegress/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_6/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_6/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity0gradients_6/ResidualRegress/add_1_grad/Reshape_18^gradients_6/ResidualRegress/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_6/ResidualRegress/MatMul_1_grad/MatMulMatMul?gradients_6/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
2gradients_6/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu?gradients_6/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_6/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp1^gradients_6/ResidualRegress/MatMul_1_grad/MatMul3^gradients_6/ResidualRegress/MatMul_1_grad/MatMul_1
ľ
Bgradients_6/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_6/ResidualRegress/MatMul_1_grad/MatMul;^gradients_6/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_6/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_6/ResidualRegress/MatMul_1_grad/MatMul_1;^gradients_6/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_6/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ç
.gradients_6/ResidualRegress/Relu_grad/ReluGradReluGradBgradients_6/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients_6/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients_6/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients_6/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_6/ResidualRegress/add_grad/Shape,gradients_6/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients_6/ResidualRegress/add_grad/SumSum.gradients_6/ResidualRegress/Relu_grad/ReluGrad:gradients_6/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients_6/ResidualRegress/add_grad/ReshapeReshape(gradients_6/ResidualRegress/add_grad/Sum*gradients_6/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients_6/ResidualRegress/add_grad/Sum_1Sum.gradients_6/ResidualRegress/Relu_grad/ReluGrad<gradients_6/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients_6/ResidualRegress/add_grad/Reshape_1Reshape*gradients_6/ResidualRegress/add_grad/Sum_1,gradients_6/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

5gradients_6/ResidualRegress/add_grad/tuple/group_depsNoOp-^gradients_6/ResidualRegress/add_grad/Reshape/^gradients_6/ResidualRegress/add_grad/Reshape_1
Ł
=gradients_6/ResidualRegress/add_grad/tuple/control_dependencyIdentity,gradients_6/ResidualRegress/add_grad/Reshape6^gradients_6/ResidualRegress/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_6/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_6/ResidualRegress/add_grad/tuple/control_dependency_1Identity.gradients_6/ResidualRegress/add_grad/Reshape_16^gradients_6/ResidualRegress/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_6/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
ě
.gradients_6/ResidualRegress/MatMul_grad/MatMulMatMul=gradients_6/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ó
0gradients_6/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1=gradients_6/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
¤
8gradients_6/ResidualRegress/MatMul_grad/tuple/group_depsNoOp/^gradients_6/ResidualRegress/MatMul_grad/MatMul1^gradients_6/ResidualRegress/MatMul_grad/MatMul_1
Ź
@gradients_6/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity.gradients_6/ResidualRegress/MatMul_grad/MatMul9^gradients_6/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_6/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ş
Bgradients_6/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity0gradients_6/ResidualRegress/MatMul_grad/MatMul_19^gradients_6/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_6/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5

beta1_power_4/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_4/AssignAssignbeta1_power_4beta1_power_4/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_4/readIdentitybeta1_power_4*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_4/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_4/AssignAssignbeta2_power_4beta2_power_4/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_4/readIdentitybeta2_power_4*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
š
=ResidualRegress/w1_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ł
3ResidualRegress/w1_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

-ResidualRegress/w1_reg/Adam/Initializer/zerosFill=ResidualRegress/w1_reg/Adam/Initializer/zeros/shape_as_tensor3ResidualRegress/w1_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ź
ResidualRegress/w1_reg/Adam
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ö
"ResidualRegress/w1_reg/Adam/AssignAssignResidualRegress/w1_reg/Adam-ResidualRegress/w1_reg/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5

 ResidualRegress/w1_reg/Adam/readIdentityResidualRegress/w1_reg/Adam*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ť
?ResidualRegress/w1_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_1/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_1/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_1
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_1/AssignAssignResidualRegress/w1_reg/Adam_1/ResidualRegress/w1_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_1/readIdentityResidualRegress/w1_reg/Adam_1*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
­
0ResidualRegress/bias1_reg/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ş
ResidualRegress/bias1_reg/Adam
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ţ
%ResidualRegress/bias1_reg/Adam/AssignAssignResidualRegress/bias1_reg/Adam0ResidualRegress/bias1_reg/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ł
#ResidualRegress/bias1_reg/Adam/readIdentityResidualRegress/bias1_reg/Adam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias1_reg/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_1/AssignAssign ResidualRegress/bias1_reg/Adam_12ResidualRegress/bias1_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_1/readIdentity ResidualRegress/bias1_reg/Adam_1*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
š
=ResidualRegress/w2_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ł
3ResidualRegress/w2_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

-ResidualRegress/w2_reg/Adam/Initializer/zerosFill=ResidualRegress/w2_reg/Adam/Initializer/zeros/shape_as_tensor3ResidualRegress/w2_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ž
ResidualRegress/w2_reg/Adam
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

÷
"ResidualRegress/w2_reg/Adam/AssignAssignResidualRegress/w2_reg/Adam-ResidualRegress/w2_reg/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:


 ResidualRegress/w2_reg/Adam/readIdentityResidualRegress/w2_reg/Adam*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ť
?ResidualRegress/w2_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_1/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_1/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_1
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_1/AssignAssignResidualRegress/w2_reg/Adam_1/ResidualRegress/w2_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_1/readIdentityResidualRegress/w2_reg/Adam_1*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

­
0ResidualRegress/bias2_reg/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ş
ResidualRegress/bias2_reg/Adam
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ţ
%ResidualRegress/bias2_reg/Adam/AssignAssignResidualRegress/bias2_reg/Adam0ResidualRegress/bias2_reg/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ł
#ResidualRegress/bias2_reg/Adam/readIdentityResidualRegress/bias2_reg/Adam*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias2_reg/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_1/AssignAssign ResidualRegress/bias2_reg/Adam_12ResidualRegress/bias2_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_1/readIdentity ResidualRegress/bias2_reg/Adam_1*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
š
=ResidualRegress/w3_reg/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ł
3ResidualRegress/w3_reg/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

-ResidualRegress/w3_reg/Adam/Initializer/zerosFill=ResidualRegress/w3_reg/Adam/Initializer/zeros/shape_as_tensor3ResidualRegress/w3_reg/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ž
ResidualRegress/w3_reg/Adam
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

÷
"ResidualRegress/w3_reg/Adam/AssignAssignResidualRegress/w3_reg/Adam-ResidualRegress/w3_reg/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:


 ResidualRegress/w3_reg/Adam/readIdentityResidualRegress/w3_reg/Adam*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ť
?ResidualRegress/w3_reg/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_1/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_1/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_1
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_1/AssignAssignResidualRegress/w3_reg/Adam_1/ResidualRegress/w3_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_1/readIdentityResidualRegress/w3_reg/Adam_1*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

­
0ResidualRegress/bias3_reg/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ş
ResidualRegress/bias3_reg/Adam
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:
ţ
%ResidualRegress/bias3_reg/Adam/AssignAssignResidualRegress/bias3_reg/Adam0ResidualRegress/bias3_reg/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ł
#ResidualRegress/bias3_reg/Adam/readIdentityResidualRegress/bias3_reg/Adam*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias3_reg/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_1/AssignAssign ResidualRegress/bias3_reg/Adam_12ResidualRegress/bias3_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_1/readIdentity ResidualRegress/bias3_reg/Adam_1*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
-ResidualRegress/w4_reg/Adam/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ź
ResidualRegress/w4_reg/Adam
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ö
"ResidualRegress/w4_reg/Adam/AssignAssignResidualRegress/w4_reg/Adam-ResidualRegress/w4_reg/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	

 ResidualRegress/w4_reg/Adam/readIdentityResidualRegress/w4_reg/Adam*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ą
/ResidualRegress/w4_reg/Adam_1/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_1
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_1/AssignAssignResidualRegress/w4_reg/Adam_1/ResidualRegress/w4_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_1/readIdentityResidualRegress/w4_reg/Adam_1*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
Ť
0ResidualRegress/bias4_reg/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
¸
ResidualRegress/bias4_reg/Adam
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:
ý
%ResidualRegress/bias4_reg/Adam/AssignAssignResidualRegress/bias4_reg/Adam0ResidualRegress/bias4_reg/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
˘
#ResidualRegress/bias4_reg/Adam/readIdentityResidualRegress/bias4_reg/Adam*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
­
2ResidualRegress/bias4_reg/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_1/AssignAssign ResidualRegress/bias4_reg/Adam_12ResidualRegress/bias4_reg/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_1/readIdentity ResidualRegress/bias4_reg/Adam_1*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_4/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_4/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_4/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_4/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
š
.Adam_4/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/AdamResidualRegress/w1_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonBgradients_6/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Á
1Adam_4/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_regResidualRegress/bias1_reg/Adam ResidualRegress/bias1_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilon?gradients_6/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:
ź
.Adam_4/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/AdamResidualRegress/w2_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonDgradients_6/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Ă
1Adam_4/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_regResidualRegress/bias2_reg/Adam ResidualRegress/bias2_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonAgradients_6/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
ź
.Adam_4/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/AdamResidualRegress/w3_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonDgradients_6/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Ă
1Adam_4/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_regResidualRegress/bias3_reg/Adam ResidualRegress/bias3_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonAgradients_6/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
ť
.Adam_4/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/AdamResidualRegress/w4_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonDgradients_6/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Â
1Adam_4/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_regResidualRegress/bias4_reg/Adam ResidualRegress/bias4_reg/Adam_1beta1_power_4/readbeta2_power_4/readAdam_4/learning_rateAdam_4/beta1Adam_4/beta2Adam_4/epsilonAgradients_6/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_4/mulMulbeta1_power_4/readAdam_4/beta12^Adam_4/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_4/AssignAssignbeta1_power_4
Adam_4/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_4/mul_1Mulbeta2_power_4/readAdam_4/beta22^Adam_4/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_4/Assign_1Assignbeta2_power_4Adam_4/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_4NoOp^Adam_4/Assign^Adam_4/Assign_12^Adam_4/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_4/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_4/update_ResidualRegress/w4_reg/ApplyAdam
T
gradients_7/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_7/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_7/FillFillgradients_7/Shapegradients_7/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
C
(gradients_7/add_24_grad/tuple/group_depsNoOp^gradients_7/Fill
ż
0gradients_7/add_24_grad/tuple/control_dependencyIdentitygradients_7/Fill)^gradients_7/add_24_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_7/Fill*
_output_shapes
: 
Á
2gradients_7/add_24_grad/tuple/control_dependency_1Identitygradients_7/Fill)^gradients_7/add_24_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_7/Fill*
_output_shapes
: 
v
%gradients_7/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ş
gradients_7/Mean_2_grad/ReshapeReshape0gradients_7/add_24_grad/tuple/control_dependency%gradients_7/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
e
gradients_7/Mean_2_grad/ShapeShapeSquare_2*
T0*
out_type0*
_output_shapes
:
¨
gradients_7/Mean_2_grad/TileTilegradients_7/Mean_2_grad/Reshapegradients_7/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_7/Mean_2_grad/Shape_1ShapeSquare_2*
T0*
out_type0*
_output_shapes
:
b
gradients_7/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_7/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_7/Mean_2_grad/ProdProdgradients_7/Mean_2_grad/Shape_1gradients_7/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
gradients_7/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_7/Mean_2_grad/Prod_1Prodgradients_7/Mean_2_grad/Shape_2gradients_7/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_7/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_7/Mean_2_grad/MaximumMaximumgradients_7/Mean_2_grad/Prod_1!gradients_7/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_7/Mean_2_grad/floordivFloorDivgradients_7/Mean_2_grad/Prodgradients_7/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_7/Mean_2_grad/CastCast gradients_7/Mean_2_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_7/Mean_2_grad/truedivRealDivgradients_7/Mean_2_grad/Tilegradients_7/Mean_2_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
'gradients_7/add_6_grad/tuple/group_depsNoOp3^gradients_7/add_24_grad/tuple/control_dependency_1
ß
/gradients_7/add_6_grad/tuple/control_dependencyIdentity2gradients_7/add_24_grad/tuple/control_dependency_1(^gradients_7/add_6_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_7/Fill*
_output_shapes
: 
á
1gradients_7/add_6_grad/tuple/control_dependency_1Identity2gradients_7/add_24_grad/tuple/control_dependency_1(^gradients_7/add_6_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_7/Fill*
_output_shapes
: 

gradients_7/Square_2_grad/ConstConst ^gradients_7/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
~
gradients_7/Square_2_grad/MulMulsub_4gradients_7/Square_2_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_7/Square_2_grad/Mul_1Mulgradients_7/Mean_2_grad/truedivgradients_7/Square_2_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
"gradients_7/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ł
gradients_7/Sum_grad/ReshapeReshape/gradients_7/add_6_grad/tuple/control_dependency"gradients_7/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
k
gradients_7/Sum_grad/ConstConst*
valueB"5      *
dtype0*
_output_shapes
:

gradients_7/Sum_grad/TileTilegradients_7/Sum_grad/Reshapegradients_7/Sum_grad/Const*

Tmultiples0*
T0*
_output_shapes
:	5
u
$gradients_7/Sum_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
š
gradients_7/Sum_1_grad/ReshapeReshape1gradients_7/add_6_grad/tuple/control_dependency_1$gradients_7/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
m
gradients_7/Sum_1_grad/ConstConst*
valueB"      *
dtype0*
_output_shapes
:

gradients_7/Sum_1_grad/TileTilegradients_7/Sum_1_grad/Reshapegradients_7/Sum_1_grad/Const*

Tmultiples0*
T0* 
_output_shapes
:

_
gradients_7/sub_4_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
u
gradients_7/sub_4_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_7/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/sub_4_grad/Shapegradients_7/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
°
gradients_7/sub_4_grad/SumSumgradients_7/Square_2_grad/Mul_1,gradients_7/sub_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_7/sub_4_grad/ReshapeReshapegradients_7/sub_4_grad/Sumgradients_7/sub_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_7/sub_4_grad/Sum_1Sumgradients_7/Square_2_grad/Mul_1.gradients_7/sub_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_7/sub_4_grad/NegNeggradients_7/sub_4_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_7/sub_4_grad/Reshape_1Reshapegradients_7/sub_4_grad/Neggradients_7/sub_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_7/sub_4_grad/tuple/group_depsNoOp^gradients_7/sub_4_grad/Reshape!^gradients_7/sub_4_grad/Reshape_1
ę
/gradients_7/sub_4_grad/tuple/control_dependencyIdentitygradients_7/sub_4_grad/Reshape(^gradients_7/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_7/sub_4_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_7/sub_4_grad/tuple/control_dependency_1Identity gradients_7/sub_4_grad/Reshape_1(^gradients_7/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_7/sub_4_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
gradients_7/Abs_grad/SignSignResidualRegress/w1_reg/read*
T0*
_output_shapes
:	5

gradients_7/Abs_grad/mulMulgradients_7/Sum_grad/Tilegradients_7/Abs_grad/Sign*
T0*
_output_shapes
:	5
k
gradients_7/Abs_1_grad/SignSignResidualRegress/w2_reg/read*
T0* 
_output_shapes
:


gradients_7/Abs_1_grad/mulMulgradients_7/Sum_1_grad/Tilegradients_7/Abs_1_grad/Sign*
T0* 
_output_shapes
:

Á
4gradients_7/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid1gradients_7/sub_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_7/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
x
.gradients_7/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_7/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_7/ResidualRegress/add_3_grad/Shape.gradients_7/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
*gradients_7/ResidualRegress/add_3_grad/SumSum4gradients_7/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients_7/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ó
.gradients_7/ResidualRegress/add_3_grad/ReshapeReshape*gradients_7/ResidualRegress/add_3_grad/Sum,gradients_7/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
,gradients_7/ResidualRegress/add_3_grad/Sum_1Sum4gradients_7/ResidualRegress/Sigmoid_grad/SigmoidGrad>gradients_7/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_7/ResidualRegress/add_3_grad/Reshape_1Reshape,gradients_7/ResidualRegress/add_3_grad/Sum_1.gradients_7/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ł
7gradients_7/ResidualRegress/add_3_grad/tuple/group_depsNoOp/^gradients_7/ResidualRegress/add_3_grad/Reshape1^gradients_7/ResidualRegress/add_3_grad/Reshape_1
Ş
?gradients_7/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity.gradients_7/ResidualRegress/add_3_grad/Reshape8^gradients_7/ResidualRegress/add_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_7/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Agradients_7/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity0gradients_7/ResidualRegress/add_3_grad/Reshape_18^gradients_7/ResidualRegress/add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ń
0gradients_7/ResidualRegress/MatMul_3_grad/MatMulMatMul?gradients_7/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
2gradients_7/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2?gradients_7/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
Ş
:gradients_7/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp1^gradients_7/ResidualRegress/MatMul_3_grad/MatMul3^gradients_7/ResidualRegress/MatMul_3_grad/MatMul_1
ľ
Bgradients_7/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity0gradients_7/ResidualRegress/MatMul_3_grad/MatMul;^gradients_7/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients_7/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity2gradients_7/ResidualRegress/MatMul_3_grad/MatMul_1;^gradients_7/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_7/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Ë
0gradients_7/ResidualRegress/Relu_2_grad/ReluGradReluGradBgradients_7/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_7/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
y
.gradients_7/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_7/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_7/ResidualRegress/add_2_grad/Shape.gradients_7/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_7/ResidualRegress/add_2_grad/SumSum0gradients_7/ResidualRegress/Relu_2_grad/ReluGrad<gradients_7/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_7/ResidualRegress/add_2_grad/ReshapeReshape*gradients_7/ResidualRegress/add_2_grad/Sum,gradients_7/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_7/ResidualRegress/add_2_grad/Sum_1Sum0gradients_7/ResidualRegress/Relu_2_grad/ReluGrad>gradients_7/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_7/ResidualRegress/add_2_grad/Reshape_1Reshape,gradients_7/ResidualRegress/add_2_grad/Sum_1.gradients_7/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_7/ResidualRegress/add_2_grad/tuple/group_depsNoOp/^gradients_7/ResidualRegress/add_2_grad/Reshape1^gradients_7/ResidualRegress/add_2_grad/Reshape_1
Ť
?gradients_7/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity.gradients_7/ResidualRegress/add_2_grad/Reshape8^gradients_7/ResidualRegress/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_7/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_7/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity0gradients_7/ResidualRegress/add_2_grad/Reshape_18^gradients_7/ResidualRegress/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_7/ResidualRegress/MatMul_2_grad/MatMulMatMul?gradients_7/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
2gradients_7/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1?gradients_7/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_7/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp1^gradients_7/ResidualRegress/MatMul_2_grad/MatMul3^gradients_7/ResidualRegress/MatMul_2_grad/MatMul_1
ľ
Bgradients_7/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_7/ResidualRegress/MatMul_2_grad/MatMul;^gradients_7/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_7/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_7/ResidualRegress/MatMul_2_grad/MatMul_1;^gradients_7/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_7/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Ë
0gradients_7/ResidualRegress/Relu_1_grad/ReluGradReluGradBgradients_7/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_7/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
y
.gradients_7/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_7/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_7/ResidualRegress/add_1_grad/Shape.gradients_7/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_7/ResidualRegress/add_1_grad/SumSum0gradients_7/ResidualRegress/Relu_1_grad/ReluGrad<gradients_7/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_7/ResidualRegress/add_1_grad/ReshapeReshape*gradients_7/ResidualRegress/add_1_grad/Sum,gradients_7/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_7/ResidualRegress/add_1_grad/Sum_1Sum0gradients_7/ResidualRegress/Relu_1_grad/ReluGrad>gradients_7/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_7/ResidualRegress/add_1_grad/Reshape_1Reshape,gradients_7/ResidualRegress/add_1_grad/Sum_1.gradients_7/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_7/ResidualRegress/add_1_grad/tuple/group_depsNoOp/^gradients_7/ResidualRegress/add_1_grad/Reshape1^gradients_7/ResidualRegress/add_1_grad/Reshape_1
Ť
?gradients_7/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity.gradients_7/ResidualRegress/add_1_grad/Reshape8^gradients_7/ResidualRegress/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_7/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_7/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity0gradients_7/ResidualRegress/add_1_grad/Reshape_18^gradients_7/ResidualRegress/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_7/ResidualRegress/MatMul_1_grad/MatMulMatMul?gradients_7/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
2gradients_7/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu?gradients_7/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_7/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp1^gradients_7/ResidualRegress/MatMul_1_grad/MatMul3^gradients_7/ResidualRegress/MatMul_1_grad/MatMul_1
ľ
Bgradients_7/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_7/ResidualRegress/MatMul_1_grad/MatMul;^gradients_7/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_7/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_7/ResidualRegress/MatMul_1_grad/MatMul_1;^gradients_7/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_7/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ç
.gradients_7/ResidualRegress/Relu_grad/ReluGradReluGradBgradients_7/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
gradients_7/AddNAddNgradients_7/Abs_1_grad/mulDgradients_7/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
T0*-
_class#
!loc:@gradients_7/Abs_1_grad/mul*
N* 
_output_shapes
:


*gradients_7/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients_7/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients_7/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_7/ResidualRegress/add_grad/Shape,gradients_7/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients_7/ResidualRegress/add_grad/SumSum.gradients_7/ResidualRegress/Relu_grad/ReluGrad:gradients_7/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients_7/ResidualRegress/add_grad/ReshapeReshape(gradients_7/ResidualRegress/add_grad/Sum*gradients_7/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients_7/ResidualRegress/add_grad/Sum_1Sum.gradients_7/ResidualRegress/Relu_grad/ReluGrad<gradients_7/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients_7/ResidualRegress/add_grad/Reshape_1Reshape*gradients_7/ResidualRegress/add_grad/Sum_1,gradients_7/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

5gradients_7/ResidualRegress/add_grad/tuple/group_depsNoOp-^gradients_7/ResidualRegress/add_grad/Reshape/^gradients_7/ResidualRegress/add_grad/Reshape_1
Ł
=gradients_7/ResidualRegress/add_grad/tuple/control_dependencyIdentity,gradients_7/ResidualRegress/add_grad/Reshape6^gradients_7/ResidualRegress/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_7/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_7/ResidualRegress/add_grad/tuple/control_dependency_1Identity.gradients_7/ResidualRegress/add_grad/Reshape_16^gradients_7/ResidualRegress/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_7/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
ě
.gradients_7/ResidualRegress/MatMul_grad/MatMulMatMul=gradients_7/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ó
0gradients_7/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1=gradients_7/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
¤
8gradients_7/ResidualRegress/MatMul_grad/tuple/group_depsNoOp/^gradients_7/ResidualRegress/MatMul_grad/MatMul1^gradients_7/ResidualRegress/MatMul_grad/MatMul_1
Ź
@gradients_7/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity.gradients_7/ResidualRegress/MatMul_grad/MatMul9^gradients_7/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_7/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ş
Bgradients_7/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity0gradients_7/ResidualRegress/MatMul_grad/MatMul_19^gradients_7/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_7/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5
Ř
gradients_7/AddN_1AddNgradients_7/Abs_grad/mulBgradients_7/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
T0*+
_class!
loc:@gradients_7/Abs_grad/mul*
N*
_output_shapes
:	5

beta1_power_5/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_5/AssignAssignbeta1_power_5beta1_power_5/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_5/readIdentitybeta1_power_5*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_5/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_5/AssignAssignbeta2_power_5beta2_power_5/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_5/readIdentitybeta2_power_5*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
ť
?ResidualRegress/w1_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_2/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_2/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_2
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_2/AssignAssignResidualRegress/w1_reg/Adam_2/ResidualRegress/w1_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_2/readIdentityResidualRegress/w1_reg/Adam_2*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ť
?ResidualRegress/w1_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_3/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_3/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_3
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_3/AssignAssignResidualRegress/w1_reg/Adam_3/ResidualRegress/w1_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_3/readIdentityResidualRegress/w1_reg/Adam_3*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
Ż
2ResidualRegress/bias1_reg/Adam_2/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_2
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_2/AssignAssign ResidualRegress/bias1_reg/Adam_22ResidualRegress/bias1_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_2/readIdentity ResidualRegress/bias1_reg/Adam_2*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias1_reg/Adam_3/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_3
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_3/AssignAssign ResidualRegress/bias1_reg/Adam_32ResidualRegress/bias1_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_3/readIdentity ResidualRegress/bias1_reg/Adam_3*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
ť
?ResidualRegress/w2_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_2/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_2/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_2
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_2/AssignAssignResidualRegress/w2_reg/Adam_2/ResidualRegress/w2_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_2/readIdentityResidualRegress/w2_reg/Adam_2*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ť
?ResidualRegress/w2_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_3/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_3/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_3
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_3/AssignAssignResidualRegress/w2_reg/Adam_3/ResidualRegress/w2_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_3/readIdentityResidualRegress/w2_reg/Adam_3*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias2_reg/Adam_2/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_2
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_2/AssignAssign ResidualRegress/bias2_reg/Adam_22ResidualRegress/bias2_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_2/readIdentity ResidualRegress/bias2_reg/Adam_2*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias2_reg/Adam_3/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_3
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_3/AssignAssign ResidualRegress/bias2_reg/Adam_32ResidualRegress/bias2_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_3/readIdentity ResidualRegress/bias2_reg/Adam_3*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
ť
?ResidualRegress/w3_reg/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_2/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_2/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_2/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_2
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_2/AssignAssignResidualRegress/w3_reg/Adam_2/ResidualRegress/w3_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_2/readIdentityResidualRegress/w3_reg/Adam_2*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ť
?ResidualRegress/w3_reg/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_3/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_3/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_3/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_3
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_3/AssignAssignResidualRegress/w3_reg/Adam_3/ResidualRegress/w3_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_3/readIdentityResidualRegress/w3_reg/Adam_3*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias3_reg/Adam_2/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_2
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_2/AssignAssign ResidualRegress/bias3_reg/Adam_22ResidualRegress/bias3_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_2/readIdentity ResidualRegress/bias3_reg/Adam_2*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias3_reg/Adam_3/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_3
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_3/AssignAssign ResidualRegress/bias3_reg/Adam_32ResidualRegress/bias3_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_3/readIdentity ResidualRegress/bias3_reg/Adam_3*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
ą
/ResidualRegress/w4_reg/Adam_2/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_2
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_2/AssignAssignResidualRegress/w4_reg/Adam_2/ResidualRegress/w4_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_2/readIdentityResidualRegress/w4_reg/Adam_2*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ą
/ResidualRegress/w4_reg/Adam_3/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_3
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_3/AssignAssignResidualRegress/w4_reg/Adam_3/ResidualRegress/w4_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_3/readIdentityResidualRegress/w4_reg/Adam_3*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
­
2ResidualRegress/bias4_reg/Adam_2/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_2
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_2/AssignAssign ResidualRegress/bias4_reg/Adam_22ResidualRegress/bias4_reg/Adam_2/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_2/readIdentity ResidualRegress/bias4_reg/Adam_2*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
­
2ResidualRegress/bias4_reg/Adam_3/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_3
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_3/AssignAssign ResidualRegress/bias4_reg/Adam_32ResidualRegress/bias4_reg/Adam_3/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_3/readIdentity ResidualRegress/bias4_reg/Adam_3*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_5/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_5/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_5/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_5/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

.Adam_5/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/Adam_2ResidualRegress/w1_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilongradients_7/AddN_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Ă
1Adam_5/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_reg ResidualRegress/bias1_reg/Adam_2 ResidualRegress/bias1_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilon?gradients_7/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:

.Adam_5/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/Adam_2ResidualRegress/w2_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilongradients_7/AddN*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_5/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_reg ResidualRegress/bias2_reg/Adam_2 ResidualRegress/bias2_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilonAgradients_7/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
ž
.Adam_5/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/Adam_2ResidualRegress/w3_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilonDgradients_7/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_5/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_reg ResidualRegress/bias3_reg/Adam_2 ResidualRegress/bias3_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilonAgradients_7/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
˝
.Adam_5/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/Adam_2ResidualRegress/w4_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilonDgradients_7/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Ä
1Adam_5/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_reg ResidualRegress/bias4_reg/Adam_2 ResidualRegress/bias4_reg/Adam_3beta1_power_5/readbeta2_power_5/readAdam_5/learning_rateAdam_5/beta1Adam_5/beta2Adam_5/epsilonAgradients_7/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_5/mulMulbeta1_power_5/readAdam_5/beta12^Adam_5/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_5/AssignAssignbeta1_power_5
Adam_5/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_5/mul_1Mulbeta2_power_5/readAdam_5/beta22^Adam_5/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_5/Assign_1Assignbeta2_power_5Adam_5/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_5NoOp^Adam_5/Assign^Adam_5/Assign_12^Adam_5/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_5/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_5/update_ResidualRegress/w4_reg/ApplyAdam
T
gradients_8/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_8/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_8/FillFillgradients_8/Shapegradients_8/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
C
(gradients_8/add_26_grad/tuple/group_depsNoOp^gradients_8/Fill
ż
0gradients_8/add_26_grad/tuple/control_dependencyIdentitygradients_8/Fill)^gradients_8/add_26_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_8/Fill*
_output_shapes
: 
Á
2gradients_8/add_26_grad/tuple/control_dependency_1Identitygradients_8/Fill)^gradients_8/add_26_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_8/Fill*
_output_shapes
: 
v
%gradients_8/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ş
gradients_8/Mean_2_grad/ReshapeReshape0gradients_8/add_26_grad/tuple/control_dependency%gradients_8/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
e
gradients_8/Mean_2_grad/ShapeShapeSquare_2*
T0*
out_type0*
_output_shapes
:
¨
gradients_8/Mean_2_grad/TileTilegradients_8/Mean_2_grad/Reshapegradients_8/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_8/Mean_2_grad/Shape_1ShapeSquare_2*
T0*
out_type0*
_output_shapes
:
b
gradients_8/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_8/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_8/Mean_2_grad/ProdProdgradients_8/Mean_2_grad/Shape_1gradients_8/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
gradients_8/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_8/Mean_2_grad/Prod_1Prodgradients_8/Mean_2_grad/Shape_2gradients_8/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_8/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_8/Mean_2_grad/MaximumMaximumgradients_8/Mean_2_grad/Prod_1!gradients_8/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_8/Mean_2_grad/floordivFloorDivgradients_8/Mean_2_grad/Prodgradients_8/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_8/Mean_2_grad/CastCast gradients_8/Mean_2_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_8/Mean_2_grad/truedivRealDivgradients_8/Mean_2_grad/Tilegradients_8/Mean_2_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
(gradients_8/add_10_grad/tuple/group_depsNoOp3^gradients_8/add_26_grad/tuple/control_dependency_1
á
0gradients_8/add_10_grad/tuple/control_dependencyIdentity2gradients_8/add_26_grad/tuple/control_dependency_1)^gradients_8/add_10_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_8/Fill*
_output_shapes
: 
ă
2gradients_8/add_10_grad/tuple/control_dependency_1Identity2gradients_8/add_26_grad/tuple/control_dependency_1)^gradients_8/add_10_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_8/Fill*
_output_shapes
: 

gradients_8/Square_2_grad/ConstConst ^gradients_8/Mean_2_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
~
gradients_8/Square_2_grad/MulMulsub_4gradients_8/Square_2_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_8/Square_2_grad/Mul_1Mulgradients_8/Mean_2_grad/truedivgradients_8/Square_2_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_8/L2Loss_grad/mulMulResidualRegress/w1_reg/read0gradients_8/add_10_grad/tuple/control_dependency*
T0*
_output_shapes
:	5
 
gradients_8/L2Loss_1_grad/mulMulResidualRegress/w2_reg/read2gradients_8/add_10_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

_
gradients_8/sub_4_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
u
gradients_8/sub_4_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_8/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_8/sub_4_grad/Shapegradients_8/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
°
gradients_8/sub_4_grad/SumSumgradients_8/Square_2_grad/Mul_1,gradients_8/sub_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_8/sub_4_grad/ReshapeReshapegradients_8/sub_4_grad/Sumgradients_8/sub_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_8/sub_4_grad/Sum_1Sumgradients_8/Square_2_grad/Mul_1.gradients_8/sub_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_8/sub_4_grad/NegNeggradients_8/sub_4_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_8/sub_4_grad/Reshape_1Reshapegradients_8/sub_4_grad/Neggradients_8/sub_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_8/sub_4_grad/tuple/group_depsNoOp^gradients_8/sub_4_grad/Reshape!^gradients_8/sub_4_grad/Reshape_1
ę
/gradients_8/sub_4_grad/tuple/control_dependencyIdentitygradients_8/sub_4_grad/Reshape(^gradients_8/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_8/sub_4_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_8/sub_4_grad/tuple/control_dependency_1Identity gradients_8/sub_4_grad/Reshape_1(^gradients_8/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_8/sub_4_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
4gradients_8/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid1gradients_8/sub_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_8/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
x
.gradients_8/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_8/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_8/ResidualRegress/add_3_grad/Shape.gradients_8/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
*gradients_8/ResidualRegress/add_3_grad/SumSum4gradients_8/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients_8/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ó
.gradients_8/ResidualRegress/add_3_grad/ReshapeReshape*gradients_8/ResidualRegress/add_3_grad/Sum,gradients_8/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
,gradients_8/ResidualRegress/add_3_grad/Sum_1Sum4gradients_8/ResidualRegress/Sigmoid_grad/SigmoidGrad>gradients_8/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_8/ResidualRegress/add_3_grad/Reshape_1Reshape,gradients_8/ResidualRegress/add_3_grad/Sum_1.gradients_8/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ł
7gradients_8/ResidualRegress/add_3_grad/tuple/group_depsNoOp/^gradients_8/ResidualRegress/add_3_grad/Reshape1^gradients_8/ResidualRegress/add_3_grad/Reshape_1
Ş
?gradients_8/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity.gradients_8/ResidualRegress/add_3_grad/Reshape8^gradients_8/ResidualRegress/add_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_8/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Agradients_8/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity0gradients_8/ResidualRegress/add_3_grad/Reshape_18^gradients_8/ResidualRegress/add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ń
0gradients_8/ResidualRegress/MatMul_3_grad/MatMulMatMul?gradients_8/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
2gradients_8/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2?gradients_8/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
Ş
:gradients_8/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp1^gradients_8/ResidualRegress/MatMul_3_grad/MatMul3^gradients_8/ResidualRegress/MatMul_3_grad/MatMul_1
ľ
Bgradients_8/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity0gradients_8/ResidualRegress/MatMul_3_grad/MatMul;^gradients_8/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients_8/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity2gradients_8/ResidualRegress/MatMul_3_grad/MatMul_1;^gradients_8/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_8/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Ë
0gradients_8/ResidualRegress/Relu_2_grad/ReluGradReluGradBgradients_8/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_8/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
y
.gradients_8/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_8/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_8/ResidualRegress/add_2_grad/Shape.gradients_8/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_8/ResidualRegress/add_2_grad/SumSum0gradients_8/ResidualRegress/Relu_2_grad/ReluGrad<gradients_8/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_8/ResidualRegress/add_2_grad/ReshapeReshape*gradients_8/ResidualRegress/add_2_grad/Sum,gradients_8/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_8/ResidualRegress/add_2_grad/Sum_1Sum0gradients_8/ResidualRegress/Relu_2_grad/ReluGrad>gradients_8/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_8/ResidualRegress/add_2_grad/Reshape_1Reshape,gradients_8/ResidualRegress/add_2_grad/Sum_1.gradients_8/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_8/ResidualRegress/add_2_grad/tuple/group_depsNoOp/^gradients_8/ResidualRegress/add_2_grad/Reshape1^gradients_8/ResidualRegress/add_2_grad/Reshape_1
Ť
?gradients_8/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity.gradients_8/ResidualRegress/add_2_grad/Reshape8^gradients_8/ResidualRegress/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_8/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_8/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity0gradients_8/ResidualRegress/add_2_grad/Reshape_18^gradients_8/ResidualRegress/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_8/ResidualRegress/MatMul_2_grad/MatMulMatMul?gradients_8/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
2gradients_8/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1?gradients_8/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_8/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp1^gradients_8/ResidualRegress/MatMul_2_grad/MatMul3^gradients_8/ResidualRegress/MatMul_2_grad/MatMul_1
ľ
Bgradients_8/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_8/ResidualRegress/MatMul_2_grad/MatMul;^gradients_8/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_8/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_8/ResidualRegress/MatMul_2_grad/MatMul_1;^gradients_8/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_8/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Ë
0gradients_8/ResidualRegress/Relu_1_grad/ReluGradReluGradBgradients_8/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_8/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
y
.gradients_8/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_8/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_8/ResidualRegress/add_1_grad/Shape.gradients_8/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_8/ResidualRegress/add_1_grad/SumSum0gradients_8/ResidualRegress/Relu_1_grad/ReluGrad<gradients_8/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_8/ResidualRegress/add_1_grad/ReshapeReshape*gradients_8/ResidualRegress/add_1_grad/Sum,gradients_8/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_8/ResidualRegress/add_1_grad/Sum_1Sum0gradients_8/ResidualRegress/Relu_1_grad/ReluGrad>gradients_8/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_8/ResidualRegress/add_1_grad/Reshape_1Reshape,gradients_8/ResidualRegress/add_1_grad/Sum_1.gradients_8/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_8/ResidualRegress/add_1_grad/tuple/group_depsNoOp/^gradients_8/ResidualRegress/add_1_grad/Reshape1^gradients_8/ResidualRegress/add_1_grad/Reshape_1
Ť
?gradients_8/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity.gradients_8/ResidualRegress/add_1_grad/Reshape8^gradients_8/ResidualRegress/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_8/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_8/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity0gradients_8/ResidualRegress/add_1_grad/Reshape_18^gradients_8/ResidualRegress/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_8/ResidualRegress/MatMul_1_grad/MatMulMatMul?gradients_8/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
2gradients_8/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu?gradients_8/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_8/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp1^gradients_8/ResidualRegress/MatMul_1_grad/MatMul3^gradients_8/ResidualRegress/MatMul_1_grad/MatMul_1
ľ
Bgradients_8/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_8/ResidualRegress/MatMul_1_grad/MatMul;^gradients_8/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_8/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_8/ResidualRegress/MatMul_1_grad/MatMul_1;^gradients_8/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_8/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ç
.gradients_8/ResidualRegress/Relu_grad/ReluGradReluGradBgradients_8/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
gradients_8/AddNAddNgradients_8/L2Loss_1_grad/mulDgradients_8/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
T0*0
_class&
$"loc:@gradients_8/L2Loss_1_grad/mul*
N* 
_output_shapes
:


*gradients_8/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients_8/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients_8/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_8/ResidualRegress/add_grad/Shape,gradients_8/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients_8/ResidualRegress/add_grad/SumSum.gradients_8/ResidualRegress/Relu_grad/ReluGrad:gradients_8/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients_8/ResidualRegress/add_grad/ReshapeReshape(gradients_8/ResidualRegress/add_grad/Sum*gradients_8/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients_8/ResidualRegress/add_grad/Sum_1Sum.gradients_8/ResidualRegress/Relu_grad/ReluGrad<gradients_8/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients_8/ResidualRegress/add_grad/Reshape_1Reshape*gradients_8/ResidualRegress/add_grad/Sum_1,gradients_8/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

5gradients_8/ResidualRegress/add_grad/tuple/group_depsNoOp-^gradients_8/ResidualRegress/add_grad/Reshape/^gradients_8/ResidualRegress/add_grad/Reshape_1
Ł
=gradients_8/ResidualRegress/add_grad/tuple/control_dependencyIdentity,gradients_8/ResidualRegress/add_grad/Reshape6^gradients_8/ResidualRegress/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_8/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_8/ResidualRegress/add_grad/tuple/control_dependency_1Identity.gradients_8/ResidualRegress/add_grad/Reshape_16^gradients_8/ResidualRegress/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_8/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
ě
.gradients_8/ResidualRegress/MatMul_grad/MatMulMatMul=gradients_8/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ó
0gradients_8/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1=gradients_8/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
¤
8gradients_8/ResidualRegress/MatMul_grad/tuple/group_depsNoOp/^gradients_8/ResidualRegress/MatMul_grad/MatMul1^gradients_8/ResidualRegress/MatMul_grad/MatMul_1
Ź
@gradients_8/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity.gradients_8/ResidualRegress/MatMul_grad/MatMul9^gradients_8/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_8/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ş
Bgradients_8/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity0gradients_8/ResidualRegress/MatMul_grad/MatMul_19^gradients_8/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_8/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5
Ţ
gradients_8/AddN_1AddNgradients_8/L2Loss_grad/mulBgradients_8/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@gradients_8/L2Loss_grad/mul*
N*
_output_shapes
:	5

beta1_power_6/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_6/AssignAssignbeta1_power_6beta1_power_6/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_6/readIdentitybeta1_power_6*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_6/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_6/AssignAssignbeta2_power_6beta2_power_6/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_6/readIdentitybeta2_power_6*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
ť
?ResidualRegress/w1_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_4/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_4/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_4
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_4/AssignAssignResidualRegress/w1_reg/Adam_4/ResidualRegress/w1_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_4/readIdentityResidualRegress/w1_reg/Adam_4*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ť
?ResidualRegress/w1_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_5/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_5/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_5
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_5/AssignAssignResidualRegress/w1_reg/Adam_5/ResidualRegress/w1_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_5/readIdentityResidualRegress/w1_reg/Adam_5*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
Ż
2ResidualRegress/bias1_reg/Adam_4/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_4/AssignAssign ResidualRegress/bias1_reg/Adam_42ResidualRegress/bias1_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_4/readIdentity ResidualRegress/bias1_reg/Adam_4*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias1_reg/Adam_5/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_5/AssignAssign ResidualRegress/bias1_reg/Adam_52ResidualRegress/bias1_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_5/readIdentity ResidualRegress/bias1_reg/Adam_5*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
ť
?ResidualRegress/w2_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_4/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_4/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_4
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_4/AssignAssignResidualRegress/w2_reg/Adam_4/ResidualRegress/w2_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_4/readIdentityResidualRegress/w2_reg/Adam_4*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ť
?ResidualRegress/w2_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_5/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_5/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_5
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_5/AssignAssignResidualRegress/w2_reg/Adam_5/ResidualRegress/w2_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_5/readIdentityResidualRegress/w2_reg/Adam_5*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias2_reg/Adam_4/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_4/AssignAssign ResidualRegress/bias2_reg/Adam_42ResidualRegress/bias2_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_4/readIdentity ResidualRegress/bias2_reg/Adam_4*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias2_reg/Adam_5/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_5/AssignAssign ResidualRegress/bias2_reg/Adam_52ResidualRegress/bias2_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_5/readIdentity ResidualRegress/bias2_reg/Adam_5*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
ť
?ResidualRegress/w3_reg/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_4/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_4/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_4/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_4
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_4/AssignAssignResidualRegress/w3_reg/Adam_4/ResidualRegress/w3_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_4/readIdentityResidualRegress/w3_reg/Adam_4*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ť
?ResidualRegress/w3_reg/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_5/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_5/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_5/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_5
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_5/AssignAssignResidualRegress/w3_reg/Adam_5/ResidualRegress/w3_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_5/readIdentityResidualRegress/w3_reg/Adam_5*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias3_reg/Adam_4/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_4/AssignAssign ResidualRegress/bias3_reg/Adam_42ResidualRegress/bias3_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_4/readIdentity ResidualRegress/bias3_reg/Adam_4*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias3_reg/Adam_5/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_5/AssignAssign ResidualRegress/bias3_reg/Adam_52ResidualRegress/bias3_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_5/readIdentity ResidualRegress/bias3_reg/Adam_5*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
ą
/ResidualRegress/w4_reg/Adam_4/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_4
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_4/AssignAssignResidualRegress/w4_reg/Adam_4/ResidualRegress/w4_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_4/readIdentityResidualRegress/w4_reg/Adam_4*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ą
/ResidualRegress/w4_reg/Adam_5/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_5
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_5/AssignAssignResidualRegress/w4_reg/Adam_5/ResidualRegress/w4_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_5/readIdentityResidualRegress/w4_reg/Adam_5*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
­
2ResidualRegress/bias4_reg/Adam_4/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_4
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_4/AssignAssign ResidualRegress/bias4_reg/Adam_42ResidualRegress/bias4_reg/Adam_4/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_4/readIdentity ResidualRegress/bias4_reg/Adam_4*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
­
2ResidualRegress/bias4_reg/Adam_5/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_5
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_5/AssignAssign ResidualRegress/bias4_reg/Adam_52ResidualRegress/bias4_reg/Adam_5/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_5/readIdentity ResidualRegress/bias4_reg/Adam_5*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_6/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_6/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_6/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_6/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

.Adam_6/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/Adam_4ResidualRegress/w1_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilongradients_8/AddN_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Ă
1Adam_6/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_reg ResidualRegress/bias1_reg/Adam_4 ResidualRegress/bias1_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilon?gradients_8/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:

.Adam_6/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/Adam_4ResidualRegress/w2_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilongradients_8/AddN*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_6/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_reg ResidualRegress/bias2_reg/Adam_4 ResidualRegress/bias2_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilonAgradients_8/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
ž
.Adam_6/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/Adam_4ResidualRegress/w3_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilonDgradients_8/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_6/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_reg ResidualRegress/bias3_reg/Adam_4 ResidualRegress/bias3_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilonAgradients_8/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
˝
.Adam_6/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/Adam_4ResidualRegress/w4_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilonDgradients_8/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Ä
1Adam_6/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_reg ResidualRegress/bias4_reg/Adam_4 ResidualRegress/bias4_reg/Adam_5beta1_power_6/readbeta2_power_6/readAdam_6/learning_rateAdam_6/beta1Adam_6/beta2Adam_6/epsilonAgradients_8/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_6/mulMulbeta1_power_6/readAdam_6/beta12^Adam_6/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_6/AssignAssignbeta1_power_6
Adam_6/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_6/mul_1Mulbeta2_power_6/readAdam_6/beta22^Adam_6/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_6/Assign_1Assignbeta2_power_6Adam_6/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_6NoOp^Adam_6/Assign^Adam_6/Assign_12^Adam_6/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_6/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_6/update_ResidualRegress/w4_reg/ApplyAdam
T
gradients_9/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_9/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_9/FillFillgradients_9/Shapegradients_9/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
%gradients_9/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients_9/Mean_4_grad/ReshapeReshapegradients_9/Fill%gradients_9/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
e
gradients_9/Mean_4_grad/ShapeShapeSquare_4*
T0*
out_type0*
_output_shapes
:
¨
gradients_9/Mean_4_grad/TileTilegradients_9/Mean_4_grad/Reshapegradients_9/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
gradients_9/Mean_4_grad/Shape_1ShapeSquare_4*
T0*
out_type0*
_output_shapes
:
b
gradients_9/Mean_4_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_9/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_9/Mean_4_grad/ProdProdgradients_9/Mean_4_grad/Shape_1gradients_9/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
gradients_9/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ś
gradients_9/Mean_4_grad/Prod_1Prodgradients_9/Mean_4_grad/Shape_2gradients_9/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_9/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_9/Mean_4_grad/MaximumMaximumgradients_9/Mean_4_grad/Prod_1!gradients_9/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_9/Mean_4_grad/floordivFloorDivgradients_9/Mean_4_grad/Prodgradients_9/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

gradients_9/Mean_4_grad/CastCast gradients_9/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients_9/Mean_4_grad/truedivRealDivgradients_9/Mean_4_grad/Tilegradients_9/Mean_4_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_9/Square_4_grad/ConstConst ^gradients_9/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
~
gradients_9/Square_4_grad/MulMulsub_6gradients_9/Square_4_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_9/Square_4_grad/Mul_1Mulgradients_9/Mean_4_grad/truedivgradients_9/Square_4_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
gradients_9/sub_6_grad/ShapeShapeTrResidual/truediv*
T0*
out_type0*
_output_shapes
:
u
gradients_9/sub_6_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ŕ
,gradients_9/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_9/sub_6_grad/Shapegradients_9/sub_6_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
°
gradients_9/sub_6_grad/SumSumgradients_9/Square_4_grad/Mul_1,gradients_9/sub_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ł
gradients_9/sub_6_grad/ReshapeReshapegradients_9/sub_6_grad/Sumgradients_9/sub_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients_9/sub_6_grad/Sum_1Sumgradients_9/Square_4_grad/Mul_1.gradients_9/sub_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
b
gradients_9/sub_6_grad/NegNeggradients_9/sub_6_grad/Sum_1*
T0*
_output_shapes
:
§
 gradients_9/sub_6_grad/Reshape_1Reshapegradients_9/sub_6_grad/Neggradients_9/sub_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_9/sub_6_grad/tuple/group_depsNoOp^gradients_9/sub_6_grad/Reshape!^gradients_9/sub_6_grad/Reshape_1
ę
/gradients_9/sub_6_grad/tuple/control_dependencyIdentitygradients_9/sub_6_grad/Reshape(^gradients_9/sub_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_9/sub_6_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
1gradients_9/sub_6_grad/tuple/control_dependency_1Identity gradients_9/sub_6_grad/Reshape_1(^gradients_9/sub_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_9/sub_6_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
4gradients_9/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid1gradients_9/sub_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_9/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
x
.gradients_9/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_9/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_9/ResidualRegress/add_3_grad/Shape.gradients_9/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
*gradients_9/ResidualRegress/add_3_grad/SumSum4gradients_9/ResidualRegress/Sigmoid_grad/SigmoidGrad<gradients_9/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ó
.gradients_9/ResidualRegress/add_3_grad/ReshapeReshape*gradients_9/ResidualRegress/add_3_grad/Sum,gradients_9/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
,gradients_9/ResidualRegress/add_3_grad/Sum_1Sum4gradients_9/ResidualRegress/Sigmoid_grad/SigmoidGrad>gradients_9/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_9/ResidualRegress/add_3_grad/Reshape_1Reshape,gradients_9/ResidualRegress/add_3_grad/Sum_1.gradients_9/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ł
7gradients_9/ResidualRegress/add_3_grad/tuple/group_depsNoOp/^gradients_9/ResidualRegress/add_3_grad/Reshape1^gradients_9/ResidualRegress/add_3_grad/Reshape_1
Ş
?gradients_9/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity.gradients_9/ResidualRegress/add_3_grad/Reshape8^gradients_9/ResidualRegress/add_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_9/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Agradients_9/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity0gradients_9/ResidualRegress/add_3_grad/Reshape_18^gradients_9/ResidualRegress/add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ń
0gradients_9/ResidualRegress/MatMul_3_grad/MatMulMatMul?gradients_9/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
2gradients_9/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2?gradients_9/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
Ş
:gradients_9/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp1^gradients_9/ResidualRegress/MatMul_3_grad/MatMul3^gradients_9/ResidualRegress/MatMul_3_grad/MatMul_1
ľ
Bgradients_9/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity0gradients_9/ResidualRegress/MatMul_3_grad/MatMul;^gradients_9/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients_9/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity2gradients_9/ResidualRegress/MatMul_3_grad/MatMul_1;^gradients_9/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_9/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Ë
0gradients_9/ResidualRegress/Relu_2_grad/ReluGradReluGradBgradients_9/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_9/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
y
.gradients_9/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_9/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_9/ResidualRegress/add_2_grad/Shape.gradients_9/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_9/ResidualRegress/add_2_grad/SumSum0gradients_9/ResidualRegress/Relu_2_grad/ReluGrad<gradients_9/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_9/ResidualRegress/add_2_grad/ReshapeReshape*gradients_9/ResidualRegress/add_2_grad/Sum,gradients_9/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_9/ResidualRegress/add_2_grad/Sum_1Sum0gradients_9/ResidualRegress/Relu_2_grad/ReluGrad>gradients_9/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_9/ResidualRegress/add_2_grad/Reshape_1Reshape,gradients_9/ResidualRegress/add_2_grad/Sum_1.gradients_9/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_9/ResidualRegress/add_2_grad/tuple/group_depsNoOp/^gradients_9/ResidualRegress/add_2_grad/Reshape1^gradients_9/ResidualRegress/add_2_grad/Reshape_1
Ť
?gradients_9/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity.gradients_9/ResidualRegress/add_2_grad/Reshape8^gradients_9/ResidualRegress/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_9/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_9/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity0gradients_9/ResidualRegress/add_2_grad/Reshape_18^gradients_9/ResidualRegress/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_9/ResidualRegress/MatMul_2_grad/MatMulMatMul?gradients_9/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
2gradients_9/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1?gradients_9/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_9/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp1^gradients_9/ResidualRegress/MatMul_2_grad/MatMul3^gradients_9/ResidualRegress/MatMul_2_grad/MatMul_1
ľ
Bgradients_9/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_9/ResidualRegress/MatMul_2_grad/MatMul;^gradients_9/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_9/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_9/ResidualRegress/MatMul_2_grad/MatMul_1;^gradients_9/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_9/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Ë
0gradients_9/ResidualRegress/Relu_1_grad/ReluGradReluGradBgradients_9/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

,gradients_9/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
y
.gradients_9/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_9/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_9/ResidualRegress/add_1_grad/Shape.gradients_9/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
á
*gradients_9/ResidualRegress/add_1_grad/SumSum0gradients_9/ResidualRegress/Relu_1_grad/ReluGrad<gradients_9/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ô
.gradients_9/ResidualRegress/add_1_grad/ReshapeReshape*gradients_9/ResidualRegress/add_1_grad/Sum,gradients_9/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
,gradients_9/ResidualRegress/add_1_grad/Sum_1Sum0gradients_9/ResidualRegress/Relu_1_grad/ReluGrad>gradients_9/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Í
0gradients_9/ResidualRegress/add_1_grad/Reshape_1Reshape,gradients_9/ResidualRegress/add_1_grad/Sum_1.gradients_9/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ł
7gradients_9/ResidualRegress/add_1_grad/tuple/group_depsNoOp/^gradients_9/ResidualRegress/add_1_grad/Reshape1^gradients_9/ResidualRegress/add_1_grad/Reshape_1
Ť
?gradients_9/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity.gradients_9/ResidualRegress/add_1_grad/Reshape8^gradients_9/ResidualRegress/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_9/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_9/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity0gradients_9/ResidualRegress/add_1_grad/Reshape_18^gradients_9/ResidualRegress/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ń
0gradients_9/ResidualRegress/MatMul_1_grad/MatMulMatMul?gradients_9/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
2gradients_9/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu?gradients_9/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

Ş
:gradients_9/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp1^gradients_9/ResidualRegress/MatMul_1_grad/MatMul3^gradients_9/ResidualRegress/MatMul_1_grad/MatMul_1
ľ
Bgradients_9/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_9/ResidualRegress/MatMul_1_grad/MatMul;^gradients_9/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Dgradients_9/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_9/ResidualRegress/MatMul_1_grad/MatMul_1;^gradients_9/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_9/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

Ç
.gradients_9/ResidualRegress/Relu_grad/ReluGradReluGradBgradients_9/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients_9/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
w
,gradients_9/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ę
:gradients_9/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_9/ResidualRegress/add_grad/Shape,gradients_9/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ű
(gradients_9/ResidualRegress/add_grad/SumSum.gradients_9/ResidualRegress/Relu_grad/ReluGrad:gradients_9/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Î
,gradients_9/ResidualRegress/add_grad/ReshapeReshape(gradients_9/ResidualRegress/add_grad/Sum*gradients_9/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
*gradients_9/ResidualRegress/add_grad/Sum_1Sum.gradients_9/ResidualRegress/Relu_grad/ReluGrad<gradients_9/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
.gradients_9/ResidualRegress/add_grad/Reshape_1Reshape*gradients_9/ResidualRegress/add_grad/Sum_1,gradients_9/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

5gradients_9/ResidualRegress/add_grad/tuple/group_depsNoOp-^gradients_9/ResidualRegress/add_grad/Reshape/^gradients_9/ResidualRegress/add_grad/Reshape_1
Ł
=gradients_9/ResidualRegress/add_grad/tuple/control_dependencyIdentity,gradients_9/ResidualRegress/add_grad/Reshape6^gradients_9/ResidualRegress/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_9/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_9/ResidualRegress/add_grad/tuple/control_dependency_1Identity.gradients_9/ResidualRegress/add_grad/Reshape_16^gradients_9/ResidualRegress/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_9/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
ě
.gradients_9/ResidualRegress/MatMul_grad/MatMulMatMul=gradients_9/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ó
0gradients_9/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1=gradients_9/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
¤
8gradients_9/ResidualRegress/MatMul_grad/tuple/group_depsNoOp/^gradients_9/ResidualRegress/MatMul_grad/MatMul1^gradients_9/ResidualRegress/MatMul_grad/MatMul_1
Ź
@gradients_9/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity.gradients_9/ResidualRegress/MatMul_grad/MatMul9^gradients_9/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_9/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ş
Bgradients_9/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity0gradients_9/ResidualRegress/MatMul_grad/MatMul_19^gradients_9/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_9/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5

beta1_power_7/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_7/AssignAssignbeta1_power_7beta1_power_7/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_7/readIdentitybeta1_power_7*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_7/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_7/AssignAssignbeta2_power_7beta2_power_7/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_7/readIdentitybeta2_power_7*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
ť
?ResidualRegress/w1_reg/Adam_6/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_6/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_6/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_6/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_6
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_6/AssignAssignResidualRegress/w1_reg/Adam_6/ResidualRegress/w1_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_6/readIdentityResidualRegress/w1_reg/Adam_6*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ť
?ResidualRegress/w1_reg/Adam_7/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_7/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_7/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_7/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_7
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_7/AssignAssignResidualRegress/w1_reg/Adam_7/ResidualRegress/w1_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_7/readIdentityResidualRegress/w1_reg/Adam_7*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
Ż
2ResidualRegress/bias1_reg/Adam_6/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_6/AssignAssign ResidualRegress/bias1_reg/Adam_62ResidualRegress/bias1_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_6/readIdentity ResidualRegress/bias1_reg/Adam_6*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias1_reg/Adam_7/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_7/AssignAssign ResidualRegress/bias1_reg/Adam_72ResidualRegress/bias1_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_7/readIdentity ResidualRegress/bias1_reg/Adam_7*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
ť
?ResidualRegress/w2_reg/Adam_6/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_6/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_6/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_6/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_6
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_6/AssignAssignResidualRegress/w2_reg/Adam_6/ResidualRegress/w2_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_6/readIdentityResidualRegress/w2_reg/Adam_6*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ť
?ResidualRegress/w2_reg/Adam_7/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_7/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_7/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_7/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_7
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_7/AssignAssignResidualRegress/w2_reg/Adam_7/ResidualRegress/w2_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_7/readIdentityResidualRegress/w2_reg/Adam_7*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias2_reg/Adam_6/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_6/AssignAssign ResidualRegress/bias2_reg/Adam_62ResidualRegress/bias2_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_6/readIdentity ResidualRegress/bias2_reg/Adam_6*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias2_reg/Adam_7/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_7/AssignAssign ResidualRegress/bias2_reg/Adam_72ResidualRegress/bias2_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_7/readIdentity ResidualRegress/bias2_reg/Adam_7*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
ť
?ResidualRegress/w3_reg/Adam_6/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_6/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_6/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_6/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_6
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_6/AssignAssignResidualRegress/w3_reg/Adam_6/ResidualRegress/w3_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_6/readIdentityResidualRegress/w3_reg/Adam_6*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ť
?ResidualRegress/w3_reg/Adam_7/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_7/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_7/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_7/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_7
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_7/AssignAssignResidualRegress/w3_reg/Adam_7/ResidualRegress/w3_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_7/readIdentityResidualRegress/w3_reg/Adam_7*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias3_reg/Adam_6/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_6/AssignAssign ResidualRegress/bias3_reg/Adam_62ResidualRegress/bias3_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_6/readIdentity ResidualRegress/bias3_reg/Adam_6*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias3_reg/Adam_7/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_7/AssignAssign ResidualRegress/bias3_reg/Adam_72ResidualRegress/bias3_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_7/readIdentity ResidualRegress/bias3_reg/Adam_7*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
ą
/ResidualRegress/w4_reg/Adam_6/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_6
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_6/AssignAssignResidualRegress/w4_reg/Adam_6/ResidualRegress/w4_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_6/readIdentityResidualRegress/w4_reg/Adam_6*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ą
/ResidualRegress/w4_reg/Adam_7/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_7
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_7/AssignAssignResidualRegress/w4_reg/Adam_7/ResidualRegress/w4_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_7/readIdentityResidualRegress/w4_reg/Adam_7*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
­
2ResidualRegress/bias4_reg/Adam_6/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_6
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_6/AssignAssign ResidualRegress/bias4_reg/Adam_62ResidualRegress/bias4_reg/Adam_6/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_6/readIdentity ResidualRegress/bias4_reg/Adam_6*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
­
2ResidualRegress/bias4_reg/Adam_7/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_7
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_7/AssignAssign ResidualRegress/bias4_reg/Adam_72ResidualRegress/bias4_reg/Adam_7/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_7/readIdentity ResidualRegress/bias4_reg/Adam_7*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_7/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_7/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_7/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_7/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ť
.Adam_7/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/Adam_6ResidualRegress/w1_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonBgradients_9/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Ă
1Adam_7/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_reg ResidualRegress/bias1_reg/Adam_6 ResidualRegress/bias1_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilon?gradients_9/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:
ž
.Adam_7/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/Adam_6ResidualRegress/w2_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonDgradients_9/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_7/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_reg ResidualRegress/bias2_reg/Adam_6 ResidualRegress/bias2_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonAgradients_9/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
ž
.Adam_7/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/Adam_6ResidualRegress/w3_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonDgradients_9/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Ĺ
1Adam_7/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_reg ResidualRegress/bias3_reg/Adam_6 ResidualRegress/bias3_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonAgradients_9/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
˝
.Adam_7/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/Adam_6ResidualRegress/w4_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonDgradients_9/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Ä
1Adam_7/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_reg ResidualRegress/bias4_reg/Adam_6 ResidualRegress/bias4_reg/Adam_7beta1_power_7/readbeta2_power_7/readAdam_7/learning_rateAdam_7/beta1Adam_7/beta2Adam_7/epsilonAgradients_9/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_7/mulMulbeta1_power_7/readAdam_7/beta12^Adam_7/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_7/AssignAssignbeta1_power_7
Adam_7/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_7/mul_1Mulbeta2_power_7/readAdam_7/beta22^Adam_7/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_7/Assign_1Assignbeta2_power_7Adam_7/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_7NoOp^Adam_7/Assign^Adam_7/Assign_12^Adam_7/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_7/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_7/update_ResidualRegress/w4_reg/ApplyAdam
U
gradients_10/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
[
gradients_10/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
gradients_10/FillFillgradients_10/Shapegradients_10/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
&gradients_10/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

 gradients_10/Mean_4_grad/ReshapeReshapegradients_10/Fill&gradients_10/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients_10/Mean_4_grad/ShapeShapeSquare_4*
T0*
out_type0*
_output_shapes
:
Ť
gradients_10/Mean_4_grad/TileTile gradients_10/Mean_4_grad/Reshapegradients_10/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
 gradients_10/Mean_4_grad/Shape_1ShapeSquare_4*
T0*
out_type0*
_output_shapes
:
c
 gradients_10/Mean_4_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
h
gradients_10/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ľ
gradients_10/Mean_4_grad/ProdProd gradients_10/Mean_4_grad/Shape_1gradients_10/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
 gradients_10/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Š
gradients_10/Mean_4_grad/Prod_1Prod gradients_10/Mean_4_grad/Shape_2 gradients_10/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
d
"gradients_10/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

 gradients_10/Mean_4_grad/MaximumMaximumgradients_10/Mean_4_grad/Prod_1"gradients_10/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

!gradients_10/Mean_4_grad/floordivFloorDivgradients_10/Mean_4_grad/Prod gradients_10/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

gradients_10/Mean_4_grad/CastCast!gradients_10/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

 gradients_10/Mean_4_grad/truedivRealDivgradients_10/Mean_4_grad/Tilegradients_10/Mean_4_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients_10/Square_4_grad/ConstConst!^gradients_10/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

gradients_10/Square_4_grad/MulMulsub_6 gradients_10/Square_4_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients_10/Square_4_grad/Mul_1Mul gradients_10/Mean_4_grad/truedivgradients_10/Square_4_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
gradients_10/sub_6_grad/ShapeShapeTrResidual/truediv*
T0*
out_type0*
_output_shapes
:
v
gradients_10/sub_6_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ă
-gradients_10/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_10/sub_6_grad/Shapegradients_10/sub_6_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ł
gradients_10/sub_6_grad/SumSum gradients_10/Square_4_grad/Mul_1-gradients_10/sub_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ś
gradients_10/sub_6_grad/ReshapeReshapegradients_10/sub_6_grad/Sumgradients_10/sub_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
gradients_10/sub_6_grad/Sum_1Sum gradients_10/Square_4_grad/Mul_1/gradients_10/sub_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
d
gradients_10/sub_6_grad/NegNeggradients_10/sub_6_grad/Sum_1*
T0*
_output_shapes
:
Ş
!gradients_10/sub_6_grad/Reshape_1Reshapegradients_10/sub_6_grad/Neggradients_10/sub_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients_10/sub_6_grad/tuple/group_depsNoOp ^gradients_10/sub_6_grad/Reshape"^gradients_10/sub_6_grad/Reshape_1
î
0gradients_10/sub_6_grad/tuple/control_dependencyIdentitygradients_10/sub_6_grad/Reshape)^gradients_10/sub_6_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_10/sub_6_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
2gradients_10/sub_6_grad/tuple/control_dependency_1Identity!gradients_10/sub_6_grad/Reshape_1)^gradients_10/sub_6_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_10/sub_6_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
5gradients_10/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid2gradients_10/sub_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_10/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
y
/gradients_10/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_10/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_10/ResidualRegress/add_3_grad/Shape/gradients_10/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
č
+gradients_10/ResidualRegress/add_3_grad/SumSum5gradients_10/ResidualRegress/Sigmoid_grad/SigmoidGrad=gradients_10/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ö
/gradients_10/ResidualRegress/add_3_grad/ReshapeReshape+gradients_10/ResidualRegress/add_3_grad/Sum-gradients_10/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
-gradients_10/ResidualRegress/add_3_grad/Sum_1Sum5gradients_10/ResidualRegress/Sigmoid_grad/SigmoidGrad?gradients_10/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ď
1gradients_10/ResidualRegress/add_3_grad/Reshape_1Reshape-gradients_10/ResidualRegress/add_3_grad/Sum_1/gradients_10/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ś
8gradients_10/ResidualRegress/add_3_grad/tuple/group_depsNoOp0^gradients_10/ResidualRegress/add_3_grad/Reshape2^gradients_10/ResidualRegress/add_3_grad/Reshape_1
Ž
@gradients_10/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity/gradients_10/ResidualRegress/add_3_grad/Reshape9^gradients_10/ResidualRegress/add_3_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_10/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Bgradients_10/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity1gradients_10/ResidualRegress/add_3_grad/Reshape_19^gradients_10/ResidualRegress/add_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ó
1gradients_10/ResidualRegress/MatMul_3_grad/MatMulMatMul@gradients_10/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
3gradients_10/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2@gradients_10/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
­
;gradients_10/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp2^gradients_10/ResidualRegress/MatMul_3_grad/MatMul4^gradients_10/ResidualRegress/MatMul_3_grad/MatMul_1
š
Cgradients_10/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity1gradients_10/ResidualRegress/MatMul_3_grad/MatMul<^gradients_10/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Egradients_10/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity3gradients_10/ResidualRegress/MatMul_3_grad/MatMul_1<^gradients_10/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_10/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Í
1gradients_10/ResidualRegress/Relu_2_grad/ReluGradReluGradCgradients_10/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_10/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
z
/gradients_10/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_10/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_10/ResidualRegress/add_2_grad/Shape/gradients_10/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
+gradients_10/ResidualRegress/add_2_grad/SumSum1gradients_10/ResidualRegress/Relu_2_grad/ReluGrad=gradients_10/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
×
/gradients_10/ResidualRegress/add_2_grad/ReshapeReshape+gradients_10/ResidualRegress/add_2_grad/Sum-gradients_10/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
-gradients_10/ResidualRegress/add_2_grad/Sum_1Sum1gradients_10/ResidualRegress/Relu_2_grad/ReluGrad?gradients_10/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Đ
1gradients_10/ResidualRegress/add_2_grad/Reshape_1Reshape-gradients_10/ResidualRegress/add_2_grad/Sum_1/gradients_10/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ś
8gradients_10/ResidualRegress/add_2_grad/tuple/group_depsNoOp0^gradients_10/ResidualRegress/add_2_grad/Reshape2^gradients_10/ResidualRegress/add_2_grad/Reshape_1
Ż
@gradients_10/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity/gradients_10/ResidualRegress/add_2_grad/Reshape9^gradients_10/ResidualRegress/add_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_10/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Bgradients_10/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity1gradients_10/ResidualRegress/add_2_grad/Reshape_19^gradients_10/ResidualRegress/add_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ó
1gradients_10/ResidualRegress/MatMul_2_grad/MatMulMatMul@gradients_10/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
3gradients_10/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1@gradients_10/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

­
;gradients_10/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp2^gradients_10/ResidualRegress/MatMul_2_grad/MatMul4^gradients_10/ResidualRegress/MatMul_2_grad/MatMul_1
š
Cgradients_10/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity1gradients_10/ResidualRegress/MatMul_2_grad/MatMul<^gradients_10/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients_10/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity3gradients_10/ResidualRegress/MatMul_2_grad/MatMul_1<^gradients_10/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_10/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Í
1gradients_10/ResidualRegress/Relu_1_grad/ReluGradReluGradCgradients_10/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_10/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
z
/gradients_10/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_10/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_10/ResidualRegress/add_1_grad/Shape/gradients_10/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
+gradients_10/ResidualRegress/add_1_grad/SumSum1gradients_10/ResidualRegress/Relu_1_grad/ReluGrad=gradients_10/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
×
/gradients_10/ResidualRegress/add_1_grad/ReshapeReshape+gradients_10/ResidualRegress/add_1_grad/Sum-gradients_10/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
-gradients_10/ResidualRegress/add_1_grad/Sum_1Sum1gradients_10/ResidualRegress/Relu_1_grad/ReluGrad?gradients_10/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Đ
1gradients_10/ResidualRegress/add_1_grad/Reshape_1Reshape-gradients_10/ResidualRegress/add_1_grad/Sum_1/gradients_10/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ś
8gradients_10/ResidualRegress/add_1_grad/tuple/group_depsNoOp0^gradients_10/ResidualRegress/add_1_grad/Reshape2^gradients_10/ResidualRegress/add_1_grad/Reshape_1
Ż
@gradients_10/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity/gradients_10/ResidualRegress/add_1_grad/Reshape9^gradients_10/ResidualRegress/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_10/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Bgradients_10/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity1gradients_10/ResidualRegress/add_1_grad/Reshape_19^gradients_10/ResidualRegress/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ó
1gradients_10/ResidualRegress/MatMul_1_grad/MatMulMatMul@gradients_10/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
3gradients_10/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu@gradients_10/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

­
;gradients_10/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp2^gradients_10/ResidualRegress/MatMul_1_grad/MatMul4^gradients_10/ResidualRegress/MatMul_1_grad/MatMul_1
š
Cgradients_10/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity1gradients_10/ResidualRegress/MatMul_1_grad/MatMul<^gradients_10/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients_10/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity3gradients_10/ResidualRegress/MatMul_1_grad/MatMul_1<^gradients_10/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_10/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

É
/gradients_10/ResidualRegress/Relu_grad/ReluGradReluGradCgradients_10/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients_10/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
x
-gradients_10/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
í
;gradients_10/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients_10/ResidualRegress/add_grad/Shape-gradients_10/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ţ
)gradients_10/ResidualRegress/add_grad/SumSum/gradients_10/ResidualRegress/Relu_grad/ReluGrad;gradients_10/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ń
-gradients_10/ResidualRegress/add_grad/ReshapeReshape)gradients_10/ResidualRegress/add_grad/Sum+gradients_10/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
+gradients_10/ResidualRegress/add_grad/Sum_1Sum/gradients_10/ResidualRegress/Relu_grad/ReluGrad=gradients_10/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ę
/gradients_10/ResidualRegress/add_grad/Reshape_1Reshape+gradients_10/ResidualRegress/add_grad/Sum_1-gradients_10/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
 
6gradients_10/ResidualRegress/add_grad/tuple/group_depsNoOp.^gradients_10/ResidualRegress/add_grad/Reshape0^gradients_10/ResidualRegress/add_grad/Reshape_1
§
>gradients_10/ResidualRegress/add_grad/tuple/control_dependencyIdentity-gradients_10/ResidualRegress/add_grad/Reshape7^gradients_10/ResidualRegress/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_10/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_10/ResidualRegress/add_grad/tuple/control_dependency_1Identity/gradients_10/ResidualRegress/add_grad/Reshape_17^gradients_10/ResidualRegress/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_10/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
î
/gradients_10/ResidualRegress/MatMul_grad/MatMulMatMul>gradients_10/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ő
1gradients_10/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1>gradients_10/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
§
9gradients_10/ResidualRegress/MatMul_grad/tuple/group_depsNoOp0^gradients_10/ResidualRegress/MatMul_grad/MatMul2^gradients_10/ResidualRegress/MatMul_grad/MatMul_1
°
Agradients_10/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity/gradients_10/ResidualRegress/MatMul_grad/MatMul:^gradients_10/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_10/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ž
Cgradients_10/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity1gradients_10/ResidualRegress/MatMul_grad/MatMul_1:^gradients_10/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_10/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5

beta1_power_8/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_8/AssignAssignbeta1_power_8beta1_power_8/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_8/readIdentitybeta1_power_8*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_8/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_8/AssignAssignbeta2_power_8beta2_power_8/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_8/readIdentitybeta2_power_8*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
ť
?ResidualRegress/w1_reg/Adam_8/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_8/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_8/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_8/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_8/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_8
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_8/AssignAssignResidualRegress/w1_reg/Adam_8/ResidualRegress/w1_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_8/readIdentityResidualRegress/w1_reg/Adam_8*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ť
?ResidualRegress/w1_reg/Adam_9/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w1_reg/Adam_9/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w1_reg/Adam_9/Initializer/zerosFill?ResidualRegress/w1_reg/Adam_9/Initializer/zeros/shape_as_tensor5ResidualRegress/w1_reg/Adam_9/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ž
ResidualRegress/w1_reg/Adam_9
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
ü
$ResidualRegress/w1_reg/Adam_9/AssignAssignResidualRegress/w1_reg/Adam_9/ResidualRegress/w1_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
˘
"ResidualRegress/w1_reg/Adam_9/readIdentityResidualRegress/w1_reg/Adam_9*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
Ż
2ResidualRegress/bias1_reg/Adam_8/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_8/AssignAssign ResidualRegress/bias1_reg/Adam_82ResidualRegress/bias1_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_8/readIdentity ResidualRegress/bias1_reg/Adam_8*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias1_reg/Adam_9/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias1_reg/Adam_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias1_reg/Adam_9/AssignAssign ResidualRegress/bias1_reg/Adam_92ResidualRegress/bias1_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias1_reg/Adam_9/readIdentity ResidualRegress/bias1_reg/Adam_9*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
ť
?ResidualRegress/w2_reg/Adam_8/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_8/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_8/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_8/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_8/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_8
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_8/AssignAssignResidualRegress/w2_reg/Adam_8/ResidualRegress/w2_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_8/readIdentityResidualRegress/w2_reg/Adam_8*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ť
?ResidualRegress/w2_reg/Adam_9/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w2_reg/Adam_9/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w2_reg/Adam_9/Initializer/zerosFill?ResidualRegress/w2_reg/Adam_9/Initializer/zeros/shape_as_tensor5ResidualRegress/w2_reg/Adam_9/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w2_reg/Adam_9
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w2_reg/Adam_9/AssignAssignResidualRegress/w2_reg/Adam_9/ResidualRegress/w2_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w2_reg/Adam_9/readIdentityResidualRegress/w2_reg/Adam_9*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias2_reg/Adam_8/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_8/AssignAssign ResidualRegress/bias2_reg/Adam_82ResidualRegress/bias2_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_8/readIdentity ResidualRegress/bias2_reg/Adam_8*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias2_reg/Adam_9/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias2_reg/Adam_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias2_reg/Adam_9/AssignAssign ResidualRegress/bias2_reg/Adam_92ResidualRegress/bias2_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias2_reg/Adam_9/readIdentity ResidualRegress/bias2_reg/Adam_9*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
ť
?ResidualRegress/w3_reg/Adam_8/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_8/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_8/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_8/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_8/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_8
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_8/AssignAssignResidualRegress/w3_reg/Adam_8/ResidualRegress/w3_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_8/readIdentityResidualRegress/w3_reg/Adam_8*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ť
?ResidualRegress/w3_reg/Adam_9/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ľ
5ResidualRegress/w3_reg/Adam_9/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

/ResidualRegress/w3_reg/Adam_9/Initializer/zerosFill?ResidualRegress/w3_reg/Adam_9/Initializer/zeros/shape_as_tensor5ResidualRegress/w3_reg/Adam_9/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ŕ
ResidualRegress/w3_reg/Adam_9
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ResidualRegress/w3_reg/Adam_9/AssignAssignResidualRegress/w3_reg/Adam_9/ResidualRegress/w3_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ł
"ResidualRegress/w3_reg/Adam_9/readIdentityResidualRegress/w3_reg/Adam_9*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Ż
2ResidualRegress/bias3_reg/Adam_8/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_8/AssignAssign ResidualRegress/bias3_reg/Adam_82ResidualRegress/bias3_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_8/readIdentity ResidualRegress/bias3_reg/Adam_8*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
Ż
2ResidualRegress/bias3_reg/Adam_9/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
ź
 ResidualRegress/bias3_reg/Adam_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

'ResidualRegress/bias3_reg/Adam_9/AssignAssign ResidualRegress/bias3_reg/Adam_92ResidualRegress/bias3_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
§
%ResidualRegress/bias3_reg/Adam_9/readIdentity ResidualRegress/bias3_reg/Adam_9*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
ą
/ResidualRegress/w4_reg/Adam_8/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_8
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_8/AssignAssignResidualRegress/w4_reg/Adam_8/ResidualRegress/w4_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_8/readIdentityResidualRegress/w4_reg/Adam_8*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
ą
/ResidualRegress/w4_reg/Adam_9/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ž
ResidualRegress/w4_reg/Adam_9
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
ü
$ResidualRegress/w4_reg/Adam_9/AssignAssignResidualRegress/w4_reg/Adam_9/ResidualRegress/w4_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˘
"ResidualRegress/w4_reg/Adam_9/readIdentityResidualRegress/w4_reg/Adam_9*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
­
2ResidualRegress/bias4_reg/Adam_8/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_8
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_8/AssignAssign ResidualRegress/bias4_reg/Adam_82ResidualRegress/bias4_reg/Adam_8/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_8/readIdentity ResidualRegress/bias4_reg/Adam_8*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
­
2ResidualRegress/bias4_reg/Adam_9/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ş
 ResidualRegress/bias4_reg/Adam_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

'ResidualRegress/bias4_reg/Adam_9/AssignAssign ResidualRegress/bias4_reg/Adam_92ResidualRegress/bias4_reg/Adam_9/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ś
%ResidualRegress/bias4_reg/Adam_9/readIdentity ResidualRegress/bias4_reg/Adam_9*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_8/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_8/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_8/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_8/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ź
.Adam_8/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/Adam_8ResidualRegress/w1_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonCgradients_10/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Ä
1Adam_8/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_reg ResidualRegress/bias1_reg/Adam_8 ResidualRegress/bias1_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilon@gradients_10/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:
ż
.Adam_8/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/Adam_8ResidualRegress/w2_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonEgradients_10/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Ć
1Adam_8/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_reg ResidualRegress/bias2_reg/Adam_8 ResidualRegress/bias2_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonBgradients_10/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
ż
.Adam_8/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/Adam_8ResidualRegress/w3_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonEgradients_10/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Ć
1Adam_8/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_reg ResidualRegress/bias3_reg/Adam_8 ResidualRegress/bias3_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonBgradients_10/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
ž
.Adam_8/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/Adam_8ResidualRegress/w4_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonEgradients_10/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Ĺ
1Adam_8/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_reg ResidualRegress/bias4_reg/Adam_8 ResidualRegress/bias4_reg/Adam_9beta1_power_8/readbeta2_power_8/readAdam_8/learning_rateAdam_8/beta1Adam_8/beta2Adam_8/epsilonBgradients_10/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_8/mulMulbeta1_power_8/readAdam_8/beta12^Adam_8/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_8/AssignAssignbeta1_power_8
Adam_8/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_8/mul_1Mulbeta2_power_8/readAdam_8/beta22^Adam_8/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_8/Assign_1Assignbeta2_power_8Adam_8/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_8NoOp^Adam_8/Assign^Adam_8/Assign_12^Adam_8/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_8/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_8/update_ResidualRegress/w4_reg/ApplyAdam
U
gradients_11/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
[
gradients_11/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
gradients_11/FillFillgradients_11/Shapegradients_11/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
&gradients_11/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

 gradients_11/Mean_4_grad/ReshapeReshapegradients_11/Fill&gradients_11/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients_11/Mean_4_grad/ShapeShapeSquare_4*
T0*
out_type0*
_output_shapes
:
Ť
gradients_11/Mean_4_grad/TileTile gradients_11/Mean_4_grad/Reshapegradients_11/Mean_4_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
 gradients_11/Mean_4_grad/Shape_1ShapeSquare_4*
T0*
out_type0*
_output_shapes
:
c
 gradients_11/Mean_4_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
h
gradients_11/Mean_4_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ľ
gradients_11/Mean_4_grad/ProdProd gradients_11/Mean_4_grad/Shape_1gradients_11/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
 gradients_11/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Š
gradients_11/Mean_4_grad/Prod_1Prod gradients_11/Mean_4_grad/Shape_2 gradients_11/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
d
"gradients_11/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

 gradients_11/Mean_4_grad/MaximumMaximumgradients_11/Mean_4_grad/Prod_1"gradients_11/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 

!gradients_11/Mean_4_grad/floordivFloorDivgradients_11/Mean_4_grad/Prod gradients_11/Mean_4_grad/Maximum*
T0*
_output_shapes
: 

gradients_11/Mean_4_grad/CastCast!gradients_11/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

 gradients_11/Mean_4_grad/truedivRealDivgradients_11/Mean_4_grad/Tilegradients_11/Mean_4_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients_11/Square_4_grad/ConstConst!^gradients_11/Mean_4_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

gradients_11/Square_4_grad/MulMulsub_6 gradients_11/Square_4_grad/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients_11/Square_4_grad/Mul_1Mul gradients_11/Mean_4_grad/truedivgradients_11/Square_4_grad/Mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
gradients_11/sub_6_grad/ShapeShapeTrResidual/truediv*
T0*
out_type0*
_output_shapes
:
v
gradients_11/sub_6_grad/Shape_1ShapeResidualRegress/Sigmoid*
T0*
out_type0*
_output_shapes
:
Ă
-gradients_11/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_11/sub_6_grad/Shapegradients_11/sub_6_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ł
gradients_11/sub_6_grad/SumSum gradients_11/Square_4_grad/Mul_1-gradients_11/sub_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ś
gradients_11/sub_6_grad/ReshapeReshapegradients_11/sub_6_grad/Sumgradients_11/sub_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
gradients_11/sub_6_grad/Sum_1Sum gradients_11/Square_4_grad/Mul_1/gradients_11/sub_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
d
gradients_11/sub_6_grad/NegNeggradients_11/sub_6_grad/Sum_1*
T0*
_output_shapes
:
Ş
!gradients_11/sub_6_grad/Reshape_1Reshapegradients_11/sub_6_grad/Neggradients_11/sub_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients_11/sub_6_grad/tuple/group_depsNoOp ^gradients_11/sub_6_grad/Reshape"^gradients_11/sub_6_grad/Reshape_1
î
0gradients_11/sub_6_grad/tuple/control_dependencyIdentitygradients_11/sub_6_grad/Reshape)^gradients_11/sub_6_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_11/sub_6_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
2gradients_11/sub_6_grad/tuple/control_dependency_1Identity!gradients_11/sub_6_grad/Reshape_1)^gradients_11/sub_6_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_11/sub_6_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
5gradients_11/ResidualRegress/Sigmoid_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid2gradients_11/sub_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_11/ResidualRegress/add_3_grad/ShapeShapeResidualRegress/MatMul_3*
T0*
out_type0*
_output_shapes
:
y
/gradients_11/ResidualRegress/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_11/ResidualRegress/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_11/ResidualRegress/add_3_grad/Shape/gradients_11/ResidualRegress/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
č
+gradients_11/ResidualRegress/add_3_grad/SumSum5gradients_11/ResidualRegress/Sigmoid_grad/SigmoidGrad=gradients_11/ResidualRegress/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ö
/gradients_11/ResidualRegress/add_3_grad/ReshapeReshape+gradients_11/ResidualRegress/add_3_grad/Sum-gradients_11/ResidualRegress/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
-gradients_11/ResidualRegress/add_3_grad/Sum_1Sum5gradients_11/ResidualRegress/Sigmoid_grad/SigmoidGrad?gradients_11/ResidualRegress/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ď
1gradients_11/ResidualRegress/add_3_grad/Reshape_1Reshape-gradients_11/ResidualRegress/add_3_grad/Sum_1/gradients_11/ResidualRegress/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ś
8gradients_11/ResidualRegress/add_3_grad/tuple/group_depsNoOp0^gradients_11/ResidualRegress/add_3_grad/Reshape2^gradients_11/ResidualRegress/add_3_grad/Reshape_1
Ž
@gradients_11/ResidualRegress/add_3_grad/tuple/control_dependencyIdentity/gradients_11/ResidualRegress/add_3_grad/Reshape9^gradients_11/ResidualRegress/add_3_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_11/ResidualRegress/add_3_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Bgradients_11/ResidualRegress/add_3_grad/tuple/control_dependency_1Identity1gradients_11/ResidualRegress/add_3_grad/Reshape_19^gradients_11/ResidualRegress/add_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/add_3_grad/Reshape_1*
_output_shapes
:
ó
1gradients_11/ResidualRegress/MatMul_3_grad/MatMulMatMul@gradients_11/ResidualRegress/add_3_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
3gradients_11/ResidualRegress/MatMul_3_grad/MatMul_1MatMulResidualRegress/Relu_2@gradients_11/ResidualRegress/add_3_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
­
;gradients_11/ResidualRegress/MatMul_3_grad/tuple/group_depsNoOp2^gradients_11/ResidualRegress/MatMul_3_grad/MatMul4^gradients_11/ResidualRegress/MatMul_3_grad/MatMul_1
š
Cgradients_11/ResidualRegress/MatMul_3_grad/tuple/control_dependencyIdentity1gradients_11/ResidualRegress/MatMul_3_grad/MatMul<^gradients_11/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Egradients_11/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1Identity3gradients_11/ResidualRegress/MatMul_3_grad/MatMul_1<^gradients_11/ResidualRegress/MatMul_3_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_11/ResidualRegress/MatMul_3_grad/MatMul_1*
_output_shapes
:	
Í
1gradients_11/ResidualRegress/Relu_2_grad/ReluGradReluGradCgradients_11/ResidualRegress/MatMul_3_grad/tuple/control_dependencyResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_11/ResidualRegress/add_2_grad/ShapeShapeResidualRegress/MatMul_2*
T0*
out_type0*
_output_shapes
:
z
/gradients_11/ResidualRegress/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_11/ResidualRegress/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_11/ResidualRegress/add_2_grad/Shape/gradients_11/ResidualRegress/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
+gradients_11/ResidualRegress/add_2_grad/SumSum1gradients_11/ResidualRegress/Relu_2_grad/ReluGrad=gradients_11/ResidualRegress/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
×
/gradients_11/ResidualRegress/add_2_grad/ReshapeReshape+gradients_11/ResidualRegress/add_2_grad/Sum-gradients_11/ResidualRegress/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
-gradients_11/ResidualRegress/add_2_grad/Sum_1Sum1gradients_11/ResidualRegress/Relu_2_grad/ReluGrad?gradients_11/ResidualRegress/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Đ
1gradients_11/ResidualRegress/add_2_grad/Reshape_1Reshape-gradients_11/ResidualRegress/add_2_grad/Sum_1/gradients_11/ResidualRegress/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ś
8gradients_11/ResidualRegress/add_2_grad/tuple/group_depsNoOp0^gradients_11/ResidualRegress/add_2_grad/Reshape2^gradients_11/ResidualRegress/add_2_grad/Reshape_1
Ż
@gradients_11/ResidualRegress/add_2_grad/tuple/control_dependencyIdentity/gradients_11/ResidualRegress/add_2_grad/Reshape9^gradients_11/ResidualRegress/add_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_11/ResidualRegress/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Bgradients_11/ResidualRegress/add_2_grad/tuple/control_dependency_1Identity1gradients_11/ResidualRegress/add_2_grad/Reshape_19^gradients_11/ResidualRegress/add_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/add_2_grad/Reshape_1*
_output_shapes	
:
ó
1gradients_11/ResidualRegress/MatMul_2_grad/MatMulMatMul@gradients_11/ResidualRegress/add_2_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
3gradients_11/ResidualRegress/MatMul_2_grad/MatMul_1MatMulResidualRegress/Relu_1@gradients_11/ResidualRegress/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

­
;gradients_11/ResidualRegress/MatMul_2_grad/tuple/group_depsNoOp2^gradients_11/ResidualRegress/MatMul_2_grad/MatMul4^gradients_11/ResidualRegress/MatMul_2_grad/MatMul_1
š
Cgradients_11/ResidualRegress/MatMul_2_grad/tuple/control_dependencyIdentity1gradients_11/ResidualRegress/MatMul_2_grad/MatMul<^gradients_11/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients_11/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1Identity3gradients_11/ResidualRegress/MatMul_2_grad/MatMul_1<^gradients_11/ResidualRegress/MatMul_2_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_11/ResidualRegress/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Í
1gradients_11/ResidualRegress/Relu_1_grad/ReluGradReluGradCgradients_11/ResidualRegress/MatMul_2_grad/tuple/control_dependencyResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients_11/ResidualRegress/add_1_grad/ShapeShapeResidualRegress/MatMul_1*
T0*
out_type0*
_output_shapes
:
z
/gradients_11/ResidualRegress/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ó
=gradients_11/ResidualRegress/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients_11/ResidualRegress/add_1_grad/Shape/gradients_11/ResidualRegress/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
+gradients_11/ResidualRegress/add_1_grad/SumSum1gradients_11/ResidualRegress/Relu_1_grad/ReluGrad=gradients_11/ResidualRegress/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
×
/gradients_11/ResidualRegress/add_1_grad/ReshapeReshape+gradients_11/ResidualRegress/add_1_grad/Sum-gradients_11/ResidualRegress/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
-gradients_11/ResidualRegress/add_1_grad/Sum_1Sum1gradients_11/ResidualRegress/Relu_1_grad/ReluGrad?gradients_11/ResidualRegress/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Đ
1gradients_11/ResidualRegress/add_1_grad/Reshape_1Reshape-gradients_11/ResidualRegress/add_1_grad/Sum_1/gradients_11/ResidualRegress/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ś
8gradients_11/ResidualRegress/add_1_grad/tuple/group_depsNoOp0^gradients_11/ResidualRegress/add_1_grad/Reshape2^gradients_11/ResidualRegress/add_1_grad/Reshape_1
Ż
@gradients_11/ResidualRegress/add_1_grad/tuple/control_dependencyIdentity/gradients_11/ResidualRegress/add_1_grad/Reshape9^gradients_11/ResidualRegress/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_11/ResidualRegress/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Bgradients_11/ResidualRegress/add_1_grad/tuple/control_dependency_1Identity1gradients_11/ResidualRegress/add_1_grad/Reshape_19^gradients_11/ResidualRegress/add_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/add_1_grad/Reshape_1*
_output_shapes	
:
ó
1gradients_11/ResidualRegress/MatMul_1_grad/MatMulMatMul@gradients_11/ResidualRegress/add_1_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
3gradients_11/ResidualRegress/MatMul_1_grad/MatMul_1MatMulResidualRegress/Relu@gradients_11/ResidualRegress/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

­
;gradients_11/ResidualRegress/MatMul_1_grad/tuple/group_depsNoOp2^gradients_11/ResidualRegress/MatMul_1_grad/MatMul4^gradients_11/ResidualRegress/MatMul_1_grad/MatMul_1
š
Cgradients_11/ResidualRegress/MatMul_1_grad/tuple/control_dependencyIdentity1gradients_11/ResidualRegress/MatMul_1_grad/MatMul<^gradients_11/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients_11/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1Identity3gradients_11/ResidualRegress/MatMul_1_grad/MatMul_1<^gradients_11/ResidualRegress/MatMul_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_11/ResidualRegress/MatMul_1_grad/MatMul_1* 
_output_shapes
:

É
/gradients_11/ResidualRegress/Relu_grad/ReluGradReluGradCgradients_11/ResidualRegress/MatMul_1_grad/tuple/control_dependencyResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients_11/ResidualRegress/add_grad/ShapeShapeResidualRegress/MatMul*
T0*
out_type0*
_output_shapes
:
x
-gradients_11/ResidualRegress/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
í
;gradients_11/ResidualRegress/add_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients_11/ResidualRegress/add_grad/Shape-gradients_11/ResidualRegress/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ţ
)gradients_11/ResidualRegress/add_grad/SumSum/gradients_11/ResidualRegress/Relu_grad/ReluGrad;gradients_11/ResidualRegress/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ń
-gradients_11/ResidualRegress/add_grad/ReshapeReshape)gradients_11/ResidualRegress/add_grad/Sum+gradients_11/ResidualRegress/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
+gradients_11/ResidualRegress/add_grad/Sum_1Sum/gradients_11/ResidualRegress/Relu_grad/ReluGrad=gradients_11/ResidualRegress/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ę
/gradients_11/ResidualRegress/add_grad/Reshape_1Reshape+gradients_11/ResidualRegress/add_grad/Sum_1-gradients_11/ResidualRegress/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
 
6gradients_11/ResidualRegress/add_grad/tuple/group_depsNoOp.^gradients_11/ResidualRegress/add_grad/Reshape0^gradients_11/ResidualRegress/add_grad/Reshape_1
§
>gradients_11/ResidualRegress/add_grad/tuple/control_dependencyIdentity-gradients_11/ResidualRegress/add_grad/Reshape7^gradients_11/ResidualRegress/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_11/ResidualRegress/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_11/ResidualRegress/add_grad/tuple/control_dependency_1Identity/gradients_11/ResidualRegress/add_grad/Reshape_17^gradients_11/ResidualRegress/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_11/ResidualRegress/add_grad/Reshape_1*
_output_shapes	
:
î
/gradients_11/ResidualRegress/MatMul_grad/MatMulMatMul>gradients_11/ResidualRegress/add_grad/tuple/control_dependencyResidualRegress/w1_reg/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ő
1gradients_11/ResidualRegress/MatMul_grad/MatMul_1MatMulconcat_1>gradients_11/ResidualRegress/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
§
9gradients_11/ResidualRegress/MatMul_grad/tuple/group_depsNoOp0^gradients_11/ResidualRegress/MatMul_grad/MatMul2^gradients_11/ResidualRegress/MatMul_grad/MatMul_1
°
Agradients_11/ResidualRegress/MatMul_grad/tuple/control_dependencyIdentity/gradients_11/ResidualRegress/MatMul_grad/MatMul:^gradients_11/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_11/ResidualRegress/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
Ž
Cgradients_11/ResidualRegress/MatMul_grad/tuple/control_dependency_1Identity1gradients_11/ResidualRegress/MatMul_grad/MatMul_1:^gradients_11/ResidualRegress/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_11/ResidualRegress/MatMul_grad/MatMul_1*
_output_shapes
:	5

beta1_power_9/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta1_power_9/AssignAssignbeta1_power_9beta1_power_9/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta1_power_9/readIdentitybeta1_power_9*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 

beta2_power_9/initial_valueConst*,
_class"
 loc:@ResidualRegress/bias1_reg*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_9
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape: *
dtype0*
_output_shapes
: 
Â
beta2_power_9/AssignAssignbeta2_power_9beta2_power_9/initial_value*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
|
beta2_power_9/readIdentitybeta2_power_9*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
ź
@ResidualRegress/w1_reg/Adam_10/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w1_reg/Adam_10/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w1_reg/Adam_10/Initializer/zerosFill@ResidualRegress/w1_reg/Adam_10/Initializer/zeros/shape_as_tensor6ResidualRegress/w1_reg/Adam_10/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ż
ResidualRegress/w1_reg/Adam_10
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
˙
%ResidualRegress/w1_reg/Adam_10/AssignAssignResidualRegress/w1_reg/Adam_100ResidualRegress/w1_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
¤
#ResidualRegress/w1_reg/Adam_10/readIdentityResidualRegress/w1_reg/Adam_10*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ź
@ResidualRegress/w1_reg/Adam_11/Initializer/zeros/shape_as_tensorConst*
valueB"5      *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w1_reg/Adam_11/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w1_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w1_reg/Adam_11/Initializer/zerosFill@ResidualRegress/w1_reg/Adam_11/Initializer/zeros/shape_as_tensor6ResidualRegress/w1_reg/Adam_11/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
ż
ResidualRegress/w1_reg/Adam_11
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w1_reg*
	container *
shape:	5*
dtype0*
_output_shapes
:	5
˙
%ResidualRegress/w1_reg/Adam_11/AssignAssignResidualRegress/w1_reg/Adam_110ResidualRegress/w1_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
¤
#ResidualRegress/w1_reg/Adam_11/readIdentityResidualRegress/w1_reg/Adam_11*
T0*)
_class
loc:@ResidualRegress/w1_reg*
_output_shapes
:	5
°
3ResidualRegress/bias1_reg/Adam_10/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias1_reg/Adam_10
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias1_reg/Adam_10/AssignAssign!ResidualRegress/bias1_reg/Adam_103ResidualRegress/bias1_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias1_reg/Adam_10/readIdentity!ResidualRegress/bias1_reg/Adam_10*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
°
3ResidualRegress/bias1_reg/Adam_11/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias1_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias1_reg/Adam_11
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias1_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias1_reg/Adam_11/AssignAssign!ResidualRegress/bias1_reg/Adam_113ResidualRegress/bias1_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias1_reg/Adam_11/readIdentity!ResidualRegress/bias1_reg/Adam_11*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes	
:
ź
@ResidualRegress/w2_reg/Adam_10/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w2_reg/Adam_10/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w2_reg/Adam_10/Initializer/zerosFill@ResidualRegress/w2_reg/Adam_10/Initializer/zeros/shape_as_tensor6ResidualRegress/w2_reg/Adam_10/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Á
ResidualRegress/w2_reg/Adam_10
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:


%ResidualRegress/w2_reg/Adam_10/AssignAssignResidualRegress/w2_reg/Adam_100ResidualRegress/w2_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ľ
#ResidualRegress/w2_reg/Adam_10/readIdentityResidualRegress/w2_reg/Adam_10*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

ź
@ResidualRegress/w2_reg/Adam_11/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w2_reg/Adam_11/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w2_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w2_reg/Adam_11/Initializer/zerosFill@ResidualRegress/w2_reg/Adam_11/Initializer/zeros/shape_as_tensor6ResidualRegress/w2_reg/Adam_11/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

Á
ResidualRegress/w2_reg/Adam_11
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w2_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:


%ResidualRegress/w2_reg/Adam_11/AssignAssignResidualRegress/w2_reg/Adam_110ResidualRegress/w2_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ľ
#ResidualRegress/w2_reg/Adam_11/readIdentityResidualRegress/w2_reg/Adam_11*
T0*)
_class
loc:@ResidualRegress/w2_reg* 
_output_shapes
:

°
3ResidualRegress/bias2_reg/Adam_10/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias2_reg/Adam_10
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias2_reg/Adam_10/AssignAssign!ResidualRegress/bias2_reg/Adam_103ResidualRegress/bias2_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias2_reg/Adam_10/readIdentity!ResidualRegress/bias2_reg/Adam_10*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
°
3ResidualRegress/bias2_reg/Adam_11/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias2_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias2_reg/Adam_11
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias2_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias2_reg/Adam_11/AssignAssign!ResidualRegress/bias2_reg/Adam_113ResidualRegress/bias2_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias2_reg/Adam_11/readIdentity!ResidualRegress/bias2_reg/Adam_11*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
_output_shapes	
:
ź
@ResidualRegress/w3_reg/Adam_10/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w3_reg/Adam_10/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w3_reg/Adam_10/Initializer/zerosFill@ResidualRegress/w3_reg/Adam_10/Initializer/zeros/shape_as_tensor6ResidualRegress/w3_reg/Adam_10/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Á
ResidualRegress/w3_reg/Adam_10
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:


%ResidualRegress/w3_reg/Adam_10/AssignAssignResidualRegress/w3_reg/Adam_100ResidualRegress/w3_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ľ
#ResidualRegress/w3_reg/Adam_10/readIdentityResidualRegress/w3_reg/Adam_10*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

ź
@ResidualRegress/w3_reg/Adam_11/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
:
Ś
6ResidualRegress/w3_reg/Adam_11/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@ResidualRegress/w3_reg*
dtype0*
_output_shapes
: 

0ResidualRegress/w3_reg/Adam_11/Initializer/zerosFill@ResidualRegress/w3_reg/Adam_11/Initializer/zeros/shape_as_tensor6ResidualRegress/w3_reg/Adam_11/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

Á
ResidualRegress/w3_reg/Adam_11
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w3_reg*
	container *
shape:
*
dtype0* 
_output_shapes
:


%ResidualRegress/w3_reg/Adam_11/AssignAssignResidualRegress/w3_reg/Adam_110ResidualRegress/w3_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ľ
#ResidualRegress/w3_reg/Adam_11/readIdentityResidualRegress/w3_reg/Adam_11*
T0*)
_class
loc:@ResidualRegress/w3_reg* 
_output_shapes
:

°
3ResidualRegress/bias3_reg/Adam_10/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias3_reg/Adam_10
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias3_reg/Adam_10/AssignAssign!ResidualRegress/bias3_reg/Adam_103ResidualRegress/bias3_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias3_reg/Adam_10/readIdentity!ResidualRegress/bias3_reg/Adam_10*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
°
3ResidualRegress/bias3_reg/Adam_11/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias3_reg*
dtype0*
_output_shapes	
:
˝
!ResidualRegress/bias3_reg/Adam_11
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias3_reg*
	container *
shape:*
dtype0*
_output_shapes	
:

(ResidualRegress/bias3_reg/Adam_11/AssignAssign!ResidualRegress/bias3_reg/Adam_113ResidualRegress/bias3_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Š
&ResidualRegress/bias3_reg/Adam_11/readIdentity!ResidualRegress/bias3_reg/Adam_11*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
_output_shapes	
:
˛
0ResidualRegress/w4_reg/Adam_10/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ż
ResidualRegress/w4_reg/Adam_10
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
˙
%ResidualRegress/w4_reg/Adam_10/AssignAssignResidualRegress/w4_reg/Adam_100ResidualRegress/w4_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
¤
#ResidualRegress/w4_reg/Adam_10/readIdentityResidualRegress/w4_reg/Adam_10*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
˛
0ResidualRegress/w4_reg/Adam_11/Initializer/zerosConst*
valueB	*    *)
_class
loc:@ResidualRegress/w4_reg*
dtype0*
_output_shapes
:	
ż
ResidualRegress/w4_reg/Adam_11
VariableV2*
shared_name *)
_class
loc:@ResidualRegress/w4_reg*
	container *
shape:	*
dtype0*
_output_shapes
:	
˙
%ResidualRegress/w4_reg/Adam_11/AssignAssignResidualRegress/w4_reg/Adam_110ResidualRegress/w4_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
¤
#ResidualRegress/w4_reg/Adam_11/readIdentityResidualRegress/w4_reg/Adam_11*
T0*)
_class
loc:@ResidualRegress/w4_reg*
_output_shapes
:	
Ž
3ResidualRegress/bias4_reg/Adam_10/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ť
!ResidualRegress/bias4_reg/Adam_10
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

(ResidualRegress/bias4_reg/Adam_10/AssignAssign!ResidualRegress/bias4_reg/Adam_103ResidualRegress/bias4_reg/Adam_10/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
¨
&ResidualRegress/bias4_reg/Adam_10/readIdentity!ResidualRegress/bias4_reg/Adam_10*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Ž
3ResidualRegress/bias4_reg/Adam_11/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ResidualRegress/bias4_reg*
dtype0*
_output_shapes
:
ť
!ResidualRegress/bias4_reg/Adam_11
VariableV2*
shared_name *,
_class"
 loc:@ResidualRegress/bias4_reg*
	container *
shape:*
dtype0*
_output_shapes
:

(ResidualRegress/bias4_reg/Adam_11/AssignAssign!ResidualRegress/bias4_reg/Adam_113ResidualRegress/bias4_reg/Adam_11/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
¨
&ResidualRegress/bias4_reg/Adam_11/readIdentity!ResidualRegress/bias4_reg/Adam_11*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
_output_shapes
:
Y
Adam_9/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_9/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_9/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_9/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ž
.Adam_9/update_ResidualRegress/w1_reg/ApplyAdam	ApplyAdamResidualRegress/w1_regResidualRegress/w1_reg/Adam_10ResidualRegress/w1_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonCgradients_11/ResidualRegress/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w1_reg*
use_nesterov( *
_output_shapes
:	5
Ć
1Adam_9/update_ResidualRegress/bias1_reg/ApplyAdam	ApplyAdamResidualRegress/bias1_reg!ResidualRegress/bias1_reg/Adam_10!ResidualRegress/bias1_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilon@gradients_11/ResidualRegress/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
use_nesterov( *
_output_shapes	
:
Á
.Adam_9/update_ResidualRegress/w2_reg/ApplyAdam	ApplyAdamResidualRegress/w2_regResidualRegress/w2_reg/Adam_10ResidualRegress/w2_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonEgradients_11/ResidualRegress/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w2_reg*
use_nesterov( * 
_output_shapes
:

Č
1Adam_9/update_ResidualRegress/bias2_reg/ApplyAdam	ApplyAdamResidualRegress/bias2_reg!ResidualRegress/bias2_reg/Adam_10!ResidualRegress/bias2_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonBgradients_11/ResidualRegress/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
use_nesterov( *
_output_shapes	
:
Á
.Adam_9/update_ResidualRegress/w3_reg/ApplyAdam	ApplyAdamResidualRegress/w3_regResidualRegress/w3_reg/Adam_10ResidualRegress/w3_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonEgradients_11/ResidualRegress/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w3_reg*
use_nesterov( * 
_output_shapes
:

Č
1Adam_9/update_ResidualRegress/bias3_reg/ApplyAdam	ApplyAdamResidualRegress/bias3_reg!ResidualRegress/bias3_reg/Adam_10!ResidualRegress/bias3_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonBgradients_11/ResidualRegress/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
use_nesterov( *
_output_shapes	
:
Ŕ
.Adam_9/update_ResidualRegress/w4_reg/ApplyAdam	ApplyAdamResidualRegress/w4_regResidualRegress/w4_reg/Adam_10ResidualRegress/w4_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonEgradients_11/ResidualRegress/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ResidualRegress/w4_reg*
use_nesterov( *
_output_shapes
:	
Ç
1Adam_9/update_ResidualRegress/bias4_reg/ApplyAdam	ApplyAdamResidualRegress/bias4_reg!ResidualRegress/bias4_reg/Adam_10!ResidualRegress/bias4_reg/Adam_11beta1_power_9/readbeta2_power_9/readAdam_9/learning_rateAdam_9/beta1Adam_9/beta2Adam_9/epsilonBgradients_11/ResidualRegress/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
use_nesterov( *
_output_shapes
:


Adam_9/mulMulbeta1_power_9/readAdam_9/beta12^Adam_9/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ş
Adam_9/AssignAssignbeta1_power_9
Adam_9/mul*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 

Adam_9/mul_1Mulbeta2_power_9/readAdam_9/beta22^Adam_9/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w4_reg/ApplyAdam*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
_output_shapes
: 
Ž
Adam_9/Assign_1Assignbeta2_power_9Adam_9/mul_1*
use_locking( *
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ä
Adam_9NoOp^Adam_9/Assign^Adam_9/Assign_12^Adam_9/update_ResidualRegress/bias1_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias2_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias3_reg/ApplyAdam2^Adam_9/update_ResidualRegress/bias4_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w1_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w2_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w3_reg/ApplyAdam/^Adam_9/update_ResidualRegress/w4_reg/ApplyAdam
U
gradients_12/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
[
gradients_12/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
gradients_12/FillFillgradients_12/Shapegradients_12/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
`
gradients_12/Max_grad/ShapeShapeAbs_8*
T0*
out_type0*
_output_shapes
:
\
gradients_12/Max_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
k
gradients_12/Max_grad/addAddConst_14gradients_12/Max_grad/Size*
T0*
_output_shapes
:

gradients_12/Max_grad/modFloorModgradients_12/Max_grad/addgradients_12/Max_grad/Size*
T0*
_output_shapes
:
g
gradients_12/Max_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
c
!gradients_12/Max_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
c
!gradients_12/Max_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ž
gradients_12/Max_grad/rangeRange!gradients_12/Max_grad/range/startgradients_12/Max_grad/Size!gradients_12/Max_grad/range/delta*

Tidx0*
_output_shapes
:
b
 gradients_12/Max_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

gradients_12/Max_grad/FillFillgradients_12/Max_grad/Shape_1 gradients_12/Max_grad/Fill/value*
T0*

index_type0*
_output_shapes
:
Ó
#gradients_12/Max_grad/DynamicStitchDynamicStitchgradients_12/Max_grad/rangegradients_12/Max_grad/modgradients_12/Max_grad/Shapegradients_12/Max_grad/Fill*
T0*
N*
_output_shapes
:
¨
gradients_12/Max_grad/ReshapeReshapeMax#gradients_12/Max_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
¸
gradients_12/Max_grad/Reshape_1Reshapegradients_12/Fill#gradients_12/Max_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

gradients_12/Max_grad/EqualEqualgradients_12/Max_grad/ReshapeAbs_8*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

gradients_12/Max_grad/CastCastgradients_12/Max_grad/Equal*

SrcT0
*
Truncate( *

DstT0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

gradients_12/Max_grad/SumSumgradients_12/Max_grad/CastConst_14*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ŕ
gradients_12/Max_grad/Reshape_2Reshapegradients_12/Max_grad/Sum#gradients_12/Max_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
gradients_12/Max_grad/divRealDivgradients_12/Max_grad/Castgradients_12/Max_grad/Reshape_2*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

gradients_12/Max_grad/mulMulgradients_12/Max_grad/divgradients_12/Max_grad/Reshape_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
c
gradients_12/Abs_8_grad/SignSignAbs_8/x*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

gradients_12/Abs_8_grad/mulMulgradients_12/Max_grad/mulgradients_12/Abs_8_grad/Sign*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2

!gradients_12/Abs_8/x_grad/unstackUnpackgradients_12/Abs_8_grad/mul*
T0*	
num*

axis *'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
x
6gradients_12/gradients/concat_1_grad/Slice_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 

7gradients_12/gradients/concat_1_grad/Slice_1_grad/ShapeShapegradients/concat_1_grad/Slice_1*
T0*
out_type0*
_output_shapes
:
{
9gradients_12/gradients/concat_1_grad/Slice_1_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
ě
7gradients_12/gradients/concat_1_grad/Slice_1_grad/stackPack6gradients_12/gradients/concat_1_grad/Slice_1_grad/Rank9gradients_12/gradients/concat_1_grad/Slice_1_grad/stack/1*
T0*

axis *
N*
_output_shapes
:
Ü
9gradients_12/gradients/concat_1_grad/Slice_1_grad/ReshapeReshape&gradients/concat_1_grad/ConcatOffset:17gradients_12/gradients/concat_1_grad/Slice_1_grad/stack*
T0*
Tshape0*
_output_shapes

:
Ľ
9gradients_12/gradients/concat_1_grad/Slice_1_grad/Shape_1Shape,gradients/ResidualRegress/MatMul_grad/MatMul*
T0*
out_type0*
_output_shapes
:
Ő
5gradients_12/gradients/concat_1_grad/Slice_1_grad/subSub9gradients_12/gradients/concat_1_grad/Slice_1_grad/Shape_17gradients_12/gradients/concat_1_grad/Slice_1_grad/Shape*
T0*
_output_shapes
:
Â
7gradients_12/gradients/concat_1_grad/Slice_1_grad/sub_1Sub5gradients_12/gradients/concat_1_grad/Slice_1_grad/sub&gradients/concat_1_grad/ConcatOffset:1*
T0*
_output_shapes
:
ď
;gradients_12/gradients/concat_1_grad/Slice_1_grad/Reshape_1Reshape7gradients_12/gradients/concat_1_grad/Slice_1_grad/sub_17gradients_12/gradients/concat_1_grad/Slice_1_grad/stack*
T0*
Tshape0*
_output_shapes

:

=gradients_12/gradients/concat_1_grad/Slice_1_grad/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
š
8gradients_12/gradients/concat_1_grad/Slice_1_grad/concatConcatV29gradients_12/gradients/concat_1_grad/Slice_1_grad/Reshape;gradients_12/gradients/concat_1_grad/Slice_1_grad/Reshape_1=gradients_12/gradients/concat_1_grad/Slice_1_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
Ü
5gradients_12/gradients/concat_1_grad/Slice_1_grad/PadPad!gradients_12/Abs_8/x_grad/unstack8gradients_12/gradients/concat_1_grad/Slice_1_grad/concat*
T0*
	Tpaddings0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙5
ü
Egradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMulMatMul5gradients_12/gradients/concat_1_grad/Slice_1_grad/PadResidualRegress/w1_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ggradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMul_1MatMul5gradients_12/gradients/concat_1_grad/Slice_1_grad/Pad*gradients/ResidualRegress/add_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	5
é
Ogradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/group_depsNoOpF^gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMulH^gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMul_1

Wgradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/control_dependencyIdentityEgradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMulP^gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/control_dependency_1IdentityGgradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMul_1P^gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/MatMul_1*
_output_shapes
:	5
ą
Bgradients_12/gradients/ResidualRegress/add_grad/Reshape_grad/ShapeShape&gradients/ResidualRegress/add_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_12/gradients/ResidualRegress/add_grad/Reshape_grad/ReshapeReshapeWgradients_12/gradients/ResidualRegress/MatMul_grad/MatMul_grad/tuple/control_dependencyBgradients_12/gradients/ResidualRegress/add_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ş
>gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/ShapeShape,gradients/ResidualRegress/Relu_grad/ReluGrad*
T0*
out_type0*
_output_shapes
:
Ň
=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/SizeConst*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
˝
<gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/addAdd8gradients/ResidualRegress/add_grad/BroadcastGradientArgs=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Size*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
<gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/modFloorMod<gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/add=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Size*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape_1Shape<gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/mod*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
Ů
Dgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/range/startConst*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Ů
Dgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/range/deltaConst*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

>gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/rangeRangeDgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/range/start=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/SizeDgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/range/delta*

Tidx0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
_output_shapes
:
Ř
Cgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Fill/valueConst*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ß
=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/FillFill@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape_1Cgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Fill/value*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
Fgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/DynamicStitchDynamicStitch>gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/range<gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/mod>gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Fill*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
Bgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Maximum/yConst*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ř
@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/MaximumMaximumFgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/DynamicStitchBgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Maximum/y*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Agradients_12/gradients/ResidualRegress/add_grad/Sum_grad/floordivFloorDiv>gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Maximum*
T0*Q
_classG
ECloc:@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/Shape*
_output_shapes
:

@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/ReshapeReshapeDgradients_12/gradients/ResidualRegress/add_grad/Reshape_grad/ReshapeFgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/TileTile@gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/ReshapeAgradients_12/gradients/ResidualRegress/add_grad/Sum_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
Ggradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ReluGradReluGrad=gradients_12/gradients/ResidualRegress/add_grad/Sum_grad/TileResidualRegress/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ShapeShapeResidualRegress/Relu*
T0*
out_type0*
_output_shapes
:

Jgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ł
Dgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zerosFillDgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ShapeJgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Ogradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/group_depsNoOpH^gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ReluGradE^gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zeros

Wgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/control_dependencyIdentityGgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ReluGradP^gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/control_dependency_1IdentityDgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zerosP^gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
Ggradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMulMatMulWgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/control_dependencyResidualRegress/w2_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Igradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMul_1MatMulWgradients_12/gradients/ResidualRegress/Relu_grad/ReluGrad_grad/tuple/control_dependency,gradients/ResidualRegress/add_1_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

ď
Qgradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/group_depsNoOpH^gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMulJ^gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMul_1

Ygradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/control_dependencyIdentityGgradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMulR^gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/control_dependency_1IdentityIgradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMul_1R^gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/MatMul_1* 
_output_shapes
:

ľ
Dgradients_12/gradients/ResidualRegress/add_1_grad/Reshape_grad/ShapeShape(gradients/ResidualRegress/add_1_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Fgradients_12/gradients/ResidualRegress/add_1_grad/Reshape_grad/ReshapeReshapeYgradients_12/gradients/ResidualRegress/MatMul_1_grad/MatMul_grad/tuple/control_dependencyDgradients_12/gradients/ResidualRegress/add_1_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ž
@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/ShapeShape.gradients/ResidualRegress/Relu_1_grad/ReluGrad*
T0*
out_type0*
_output_shapes
:
Ö
?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/SizeConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ĺ
>gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/addAdd:gradients/ResidualRegress/add_1_grad/BroadcastGradientArgs?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/modFloorMod>gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/add?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Bgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape_1Shape>gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/mod*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
Ý
Fgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/range/startConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Ý
Fgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/range/deltaConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/rangeRangeFgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/range/start?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/SizeFgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/range/delta*

Tidx0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
_output_shapes
:
Ü
Egradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Fill/valueConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ç
?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/FillFillBgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape_1Egradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Fill/value*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
Hgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/DynamicStitchDynamicStitch@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/range>gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/mod@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Fill*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
Dgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Maximum/yConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ŕ
Bgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/MaximumMaximumHgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/DynamicStitchDgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Maximum/y*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
Cgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/floordivFloorDiv@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/ShapeBgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Maximum*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/Shape*
_output_shapes
:

Bgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/ReshapeReshapeFgradients_12/gradients/ResidualRegress/add_1_grad/Reshape_grad/ReshapeHgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/TileTileBgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/ReshapeCgradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Igradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ReluGradReluGrad?gradients_12/gradients/ResidualRegress/add_1_grad/Sum_grad/TileResidualRegress/Relu_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Fgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ShapeShapeResidualRegress/Relu_1*
T0*
out_type0*
_output_shapes
:

Lgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
Fgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zerosFillFgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ShapeLgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Qgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/group_depsNoOpJ^gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ReluGradG^gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zeros

Ygradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/control_dependencyIdentityIgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ReluGradR^gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/control_dependency_1IdentityFgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zerosR^gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Ggradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMulMatMulYgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/control_dependencyResidualRegress/w3_reg/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Igradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMul_1MatMulYgradients_12/gradients/ResidualRegress/Relu_1_grad/ReluGrad_grad/tuple/control_dependency,gradients/ResidualRegress/add_2_grad/Reshape*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:

ď
Qgradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/group_depsNoOpH^gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMulJ^gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMul_1

Ygradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/control_dependencyIdentityGgradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMulR^gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/control_dependency_1IdentityIgradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMul_1R^gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/MatMul_1* 
_output_shapes
:

ľ
Dgradients_12/gradients/ResidualRegress/add_2_grad/Reshape_grad/ShapeShape(gradients/ResidualRegress/add_2_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Fgradients_12/gradients/ResidualRegress/add_2_grad/Reshape_grad/ReshapeReshapeYgradients_12/gradients/ResidualRegress/MatMul_2_grad/MatMul_grad/tuple/control_dependencyDgradients_12/gradients/ResidualRegress/add_2_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ž
@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/ShapeShape.gradients/ResidualRegress/Relu_2_grad/ReluGrad*
T0*
out_type0*
_output_shapes
:
Ö
?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/SizeConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ĺ
>gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/addAdd:gradients/ResidualRegress/add_2_grad/BroadcastGradientArgs?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/modFloorMod>gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/add?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Bgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape_1Shape>gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/mod*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
Ý
Fgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/range/startConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Ý
Fgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/range/deltaConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/rangeRangeFgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/range/start?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/SizeFgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/range/delta*

Tidx0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
_output_shapes
:
Ü
Egradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Fill/valueConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ç
?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/FillFillBgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape_1Egradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Fill/value*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
Hgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/DynamicStitchDynamicStitch@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/range>gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/mod@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Fill*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
Dgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Maximum/yConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ŕ
Bgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/MaximumMaximumHgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/DynamicStitchDgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Maximum/y*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
Cgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/floordivFloorDiv@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/ShapeBgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Maximum*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/Shape*
_output_shapes
:

Bgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/ReshapeReshapeFgradients_12/gradients/ResidualRegress/add_2_grad/Reshape_grad/ReshapeHgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/TileTileBgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/ReshapeCgradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Igradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ReluGradReluGrad?gradients_12/gradients/ResidualRegress/add_2_grad/Sum_grad/TileResidualRegress/Relu_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Fgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ShapeShapeResidualRegress/Relu_2*
T0*
out_type0*
_output_shapes
:

Lgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Š
Fgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zerosFillFgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ShapeLgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Qgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/group_depsNoOpJ^gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ReluGradG^gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zeros

Ygradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/control_dependencyIdentityIgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ReluGradR^gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/control_dependency_1IdentityFgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zerosR^gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
Ggradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMulMatMulYgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/control_dependencyResidualRegress/w4_reg/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Igradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMul_1MatMulYgradients_12/gradients/ResidualRegress/Relu_2_grad/ReluGrad_grad/tuple/control_dependency,gradients/ResidualRegress/add_3_grad/Reshape*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	
ď
Qgradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/group_depsNoOpH^gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMulJ^gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMul_1

Ygradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/control_dependencyIdentityGgradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMulR^gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/control_dependency_1IdentityIgradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMul_1R^gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/MatMul_1*
_output_shapes
:	
ľ
Dgradients_12/gradients/ResidualRegress/add_3_grad/Reshape_grad/ShapeShape(gradients/ResidualRegress/add_3_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Fgradients_12/gradients/ResidualRegress/add_3_grad/Reshape_grad/ReshapeReshapeYgradients_12/gradients/ResidualRegress/MatMul_3_grad/MatMul_grad/tuple/control_dependencyDgradients_12/gradients/ResidualRegress/add_3_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
˛
@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/ShapeShape2gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad*
T0*
out_type0*
_output_shapes
:
Ö
?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/SizeConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ĺ
>gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/addAdd:gradients/ResidualRegress/add_3_grad/BroadcastGradientArgs?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
>gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/modFloorMod>gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/add?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Size*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Bgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape_1Shape>gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/mod*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
Ý
Fgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/range/startConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Ý
Fgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/range/deltaConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/rangeRangeFgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/range/start?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/SizeFgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/range/delta*

Tidx0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
_output_shapes
:
Ü
Egradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Fill/valueConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ç
?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/FillFillBgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape_1Egradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Fill/value*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
Hgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/DynamicStitchDynamicStitch@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/range>gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/mod@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Fill*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
Dgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Maximum/yConst*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ŕ
Bgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/MaximumMaximumHgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/DynamicStitchDgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Maximum/y*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
Cgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/floordivFloorDiv@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/ShapeBgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Maximum*
T0*S
_classI
GEloc:@gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Shape*
_output_shapes
:

Bgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/ReshapeReshapeFgradients_12/gradients/ResidualRegress/add_3_grad/Reshape_grad/ReshapeHgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/TileTileBgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/ReshapeCgradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
Hgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mulMul?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Tilegradients/AddN*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Lgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_1/xConst@^gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 

Jgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_1MulLgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_1/xHgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Jgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_2MulJgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_1ResidualRegress/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/subSubHgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mulJgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/mul_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Pgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGradSigmoidGradResidualRegress/Sigmoid?gradients_12/gradients/ResidualRegress/add_3_grad/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
Ugradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/group_depsNoOpQ^gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGradI^gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/sub

]gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/control_dependencyIdentityHgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/subV^gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
_gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/control_dependency_1IdentityPgradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGradV^gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients_12/gradients/AddN_grad/tuple/group_depsNoOp`^gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/control_dependency_1
ń
9gradients_12/gradients/AddN_grad/tuple/control_dependencyIdentity_gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/control_dependency_12^gradients_12/gradients/AddN_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
;gradients_12/gradients/AddN_grad/tuple/control_dependency_1Identity_gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/tuple/control_dependency_12^gradients_12/gradients/AddN_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_12/gradients/ResidualRegress/Sigmoid_grad/SigmoidGrad_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_grad/ShapeShape)gradients/TrResidual_2/truediv_1_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ggradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_grad/ReshapeReshape9gradients_12/gradients/AddN_grad/tuple/control_dependencyEgradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
­
Agradients_12/gradients/TrResidual_2/sub_grad/Reshape_1_grad/ShapeShape#gradients/TrResidual_2/sub_grad/Neg*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Cgradients_12/gradients/TrResidual_2/sub_grad/Reshape_1_grad/ReshapeReshape;gradients_12/gradients/AddN_grad/tuple/control_dependency_1Agradients_12/gradients/TrResidual_2/sub_grad/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ž
Agradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/ShapeShape-gradients/TrResidual_2/truediv_1_grad/RealDiv*
T0*
out_type0*
_output_shapes
:
Ř
@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/SizeConst*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
É
?gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/addAdd;gradients/TrResidual_2/truediv_1_grad/BroadcastGradientArgs@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Size*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
?gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/modFloorMod?gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/add@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Size*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape_1Shape?gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/mod*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
ß
Ggradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/range/startConst*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
ß
Ggradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/range/deltaConst*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

Agradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/rangeRangeGgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/range/start@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/SizeGgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/range/delta*

Tidx0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
_output_shapes
:
Ţ
Fgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Fill/valueConst*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ë
@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/FillFillCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape_1Fgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Fill/value*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
Igradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/DynamicStitchDynamicStitchAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/range?gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/modAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Fill*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
Egradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Maximum/yConst*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ä
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/MaximumMaximumIgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/DynamicStitchEgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Maximum/y*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Dgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/floordivFloorDivAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/ShapeCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Maximum*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/Shape*
_output_shapes
:

Cgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/ReshapeReshapeGgradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_grad/ReshapeIgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/TileTileCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/ReshapeDgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
9gradients_12/gradients/TrResidual_2/sub_grad/Neg_grad/NegNegCgradients_12/gradients/TrResidual_2/sub_grad/Reshape_1_grad/Reshape*
T0*
_output_shapes
:
¨
Egradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/ShapeShape#gradients/TrResidual_2/Log_grad/mul*
T0*
out_type0*
_output_shapes
:

Ggradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Shape_1ShapeTrResidual_2/add*
T0*
out_type0*
_output_shapes
:
ť
Ugradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/ShapeGgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ř
Ggradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDivRealDiv@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/TileTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/SumSumGgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDivUgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

Ggradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/ReshapeReshapeCgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/SumEgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/NegNeg#gradients/TrResidual_2/Log_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
Igradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDiv_1RealDivCgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/NegTrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
Igradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDiv_2RealDivIgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDiv_1TrResidual_2/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/mulMul@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_grad/TileIgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Egradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Sum_1SumCgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/mulWgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
¤
Igradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape_1ReshapeEgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Sum_1Ggradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Pgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/group_depsNoOpH^gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/ReshapeJ^gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape_1

Xgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/control_dependencyIdentityGgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/ReshapeQ^gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Zgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/control_dependency_1IdentityIgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape_1Q^gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
=gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/ShapeShape'gradients/TrResidual_2/add_grad/Reshape*
T0*
out_type0*
_output_shapes
:
Đ
<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/SizeConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
š
;gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/addAdd7gradients/TrResidual_2/sub_grad/BroadcastGradientArgs:1<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Size*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
;gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/modFloorMod;gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/add<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Size*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape_1Shape;gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/mod*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
out_type0*
_output_shapes
:
×
Cgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/range/startConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
×
Cgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/range/deltaConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

=gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/rangeRangeCgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/range/start<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/SizeCgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/range/delta*

Tidx0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
_output_shapes
:
Ö
Bgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Fill/valueConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ű
<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/FillFill?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape_1Bgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Fill/value*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
Egradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/DynamicStitchDynamicStitch=gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/range;gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/mod=gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Fill*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Agradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Maximum/yConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ô
?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/MaximumMaximumEgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/DynamicStitchAgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Maximum/y*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/floordivFloorDiv=gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Maximum*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Shape*
_output_shapes
:
ý
?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/ReshapeReshape9gradients_12/gradients/TrResidual_2/sub_grad/Neg_grad/NegEgradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/TileTile?gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Reshape@gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
?gradients_12/gradients/TrResidual_2/add_grad/Reshape_grad/ShapeShape#gradients/TrResidual_2/add_grad/Sum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
Agradients_12/gradients/TrResidual_2/add_grad/Reshape_grad/ReshapeReshape<gradients_12/gradients/TrResidual_2/sub_grad/Sum_1_grad/Tile?gradients_12/gradients/TrResidual_2/add_grad/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ş
;gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/ShapeShape/gradients/TrResidual_2/truediv_1_grad/Reshape_1*
T0*
out_type0*
_output_shapes
:
Ě
:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/SizeConst*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ą
9gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/addAdd5gradients/TrResidual_2/add_grad/BroadcastGradientArgs:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Size*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
9gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/modFloorMod9gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/add:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Size*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape_1Shape9gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/mod*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
out_type0*
_output_shapes
:
Ó
Agradients_12/gradients/TrResidual_2/add_grad/Sum_grad/range/startConst*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Ó
Agradients_12/gradients/TrResidual_2/add_grad/Sum_grad/range/deltaConst*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ţ
;gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/rangeRangeAgradients_12/gradients/TrResidual_2/add_grad/Sum_grad/range/start:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/SizeAgradients_12/gradients/TrResidual_2/add_grad/Sum_grad/range/delta*

Tidx0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
_output_shapes
:
Ň
@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Fill/valueConst*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ó
:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/FillFill=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape_1@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Fill/value*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
Cgradients_12/gradients/TrResidual_2/add_grad/Sum_grad/DynamicStitchDynamicStitch;gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/range9gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/mod;gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Fill*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Maximum/yConst*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ě
=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/MaximumMaximumCgradients_12/gradients/TrResidual_2/add_grad/Sum_grad/DynamicStitch?gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Maximum/y*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
>gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/floordivFloorDiv;gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Maximum*
T0*N
_classD
B@loc:@gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Shape*
_output_shapes
:

=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/ReshapeReshapeAgradients_12/gradients/TrResidual_2/add_grad/Reshape_grad/ReshapeCgradients_12/gradients/TrResidual_2/add_grad/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/TileTile=gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/Reshape>gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Ggradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_1_grad/ShapeShape+gradients/TrResidual_2/truediv_1_grad/Sum_1*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Igradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_1_grad/ReshapeReshape:gradients_12/gradients/TrResidual_2/add_grad/Sum_grad/TileGgradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ź
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/ShapeShape)gradients/TrResidual_2/truediv_1_grad/mul*
T0*
out_type0*
_output_shapes
:
Ü
Bgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/SizeConst*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ń
Agradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/addAdd=gradients/TrResidual_2/truediv_1_grad/BroadcastGradientArgs:1Bgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Size*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ú
Agradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/modFloorModAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/addBgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Size*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Egradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape_1ShapeAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/mod*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
out_type0*
_output_shapes
:
ă
Igradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/range/startConst*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
ă
Igradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/range/deltaConst*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ś
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/rangeRangeIgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/range/startBgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/SizeIgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/range/delta*

Tidx0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
_output_shapes
:
â
Hgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Fill/valueConst*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ó
Bgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/FillFillEgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape_1Hgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Fill/value*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
Kgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/DynamicStitchDynamicStitchCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/rangeAgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/modCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/ShapeBgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Fill*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Ggradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Maximum/yConst*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ě
Egradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/MaximumMaximumKgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/DynamicStitchGgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Maximum/y*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
Fgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/floordivFloorDivCgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/ShapeEgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Maximum*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Shape*
_output_shapes
:

Egradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/ReshapeReshapeIgradients_12/gradients/TrResidual_2/truediv_1_grad/Reshape_1_grad/ReshapeKgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

Bgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/TileTileEgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/ReshapeFgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Agradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/ShapeShape#gradients/TrResidual_2/Log_grad/mul*
T0*
out_type0*
_output_shapes
:
˛
Cgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Shape_1Shape/gradients/TrResidual_2/truediv_1_grad/RealDiv_2*
T0*
out_type0*
_output_shapes
:
Ż
Qgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/ShapeCgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
í
?gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/MulMulBgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Tile/gradients/TrResidual_2/truediv_1_grad/RealDiv_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/SumSum?gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/MulQgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

Cgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/ReshapeReshape?gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/SumAgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
Agradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Mul_1Mul#gradients/TrResidual_2/Log_grad/mulBgradients_12/gradients/TrResidual_2/truediv_1_grad/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
Agradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Sum_1SumAgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Mul_1Sgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

Egradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Reshape_1ReshapeAgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Sum_1Cgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Lgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/group_depsNoOpD^gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/ReshapeF^gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Reshape_1
ţ
Tgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/control_dependencyIdentityCgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/ReshapeM^gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/control_dependency_1IdentityEgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Reshape_1M^gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
gradients_12/AddNAddNXgradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/tuple/control_dependencyTgradients_12/gradients/TrResidual_2/truediv_1_grad/mul_grad/tuple/control_dependency*
T0*Z
_classP
NLloc:@gradients_12/gradients/TrResidual_2/truediv_1_grad/RealDiv_grad/Reshape*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/ShapeShape)gradients/TrResidual_2/mul_grad/Reshape_1*
T0*
out_type0*
_output_shapes
:
§
=gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Shape_1Shape*gradients/TrResidual_2/Log_grad/Reciprocal*
T0*
out_type0*
_output_shapes
:

Kgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Shape=gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
9gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/MulMulgradients_12/AddN*gradients/TrResidual_2/Log_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/SumSum9gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/MulKgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

=gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/ReshapeReshape9gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Sum;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Mul_1Mul)gradients/TrResidual_2/mul_grad/Reshape_1gradients_12/AddN*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Sum_1Sum;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Mul_1Mgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

?gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape_1Reshape;gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Sum_1=gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Fgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/group_depsNoOp>^gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape@^gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape_1
ć
Ngradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/control_dependencyIdentity=gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/ReshapeG^gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
Pgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/control_dependency_1Identity?gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape_1G^gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_12/gradients/TrResidual_2/Log_grad/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Agradients_12/gradients/TrResidual_2/mul_grad/Reshape_1_grad/ShapeShape%gradients/TrResidual_2/mul_grad/Sum_1*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cgradients_12/gradients/TrResidual_2/mul_grad/Reshape_1_grad/ReshapeReshapeNgradients_12/gradients/TrResidual_2/Log_grad/mul_grad/tuple/control_dependencyAgradients_12/gradients/TrResidual_2/mul_grad/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
˘
=gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/ShapeShape%gradients/TrResidual_2/mul_grad/Mul_1*
T0*
out_type0*
_output_shapes
:
Đ
<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/SizeConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
š
;gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/addAdd7gradients/TrResidual_2/mul_grad/BroadcastGradientArgs:1<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Size*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
;gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/modFloorMod;gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/add<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Size*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape_1Shape;gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/mod*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
out_type0*
_output_shapes
:
×
Cgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/range/startConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
×
Cgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/range/deltaConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

=gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/rangeRangeCgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/range/start<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/SizeCgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/range/delta*

Tidx0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
_output_shapes
:
Ö
Bgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Fill/valueConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ű
<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/FillFill?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape_1Bgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Fill/value*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
Egradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/DynamicStitchDynamicStitch=gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/range;gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/mod=gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Fill*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Agradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Maximum/yConst*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ô
?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/MaximumMaximumEgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/DynamicStitchAgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Maximum/y*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/floordivFloorDiv=gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Maximum*
T0*P
_classF
DBloc:@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Shape*
_output_shapes
:

?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/ReshapeReshapeCgradients_12/gradients/TrResidual_2/mul_grad/Reshape_1_grad/ReshapeEgradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/TileTile?gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Reshape@gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

?gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Shape_1Shapegradients/add_14_grad/Reshape_1*
T0*
out_type0*
_output_shapes
:
Ł
Mgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Shape?gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ó
;gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/MulMul<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Tilegradients/add_14_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/SumSum;gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/MulMgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ů
?gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/ReshapeReshape;gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Sum=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Ę
=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Mul_1MulTrResidual_2/truediv<gradients_12/gradients/TrResidual_2/mul_grad/Sum_1_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Sum_1Sum=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Mul_1Ogradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

Agradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Reshape_1Reshape=gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Sum_1?gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
Hgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/group_depsNoOp@^gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/ReshapeB^gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Reshape_1
á
Pgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/control_dependencyIdentity?gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/ReshapeI^gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Reshape*
_output_shapes
:
ô
Rgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/control_dependency_1IdentityAgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Reshape_1I^gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
,gradients_12/TrResidual_2/truediv_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
.gradients_12/TrResidual_2/truediv_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
đ
<gradients_12/TrResidual_2/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_12/TrResidual_2/truediv_grad/Shape.gradients_12/TrResidual_2/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ç
.gradients_12/TrResidual_2/truediv_grad/RealDivRealDivPgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/control_dependencyTrResidual/alpha/read*
T0*
_output_shapes
:
Ý
*gradients_12/TrResidual_2/truediv_grad/SumSum.gradients_12/TrResidual_2/truediv_grad/RealDiv<gradients_12/TrResidual_2/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Â
.gradients_12/TrResidual_2/truediv_grad/ReshapeReshape*gradients_12/TrResidual_2/truediv_grad/Sum,gradients_12/TrResidual_2/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
j
*gradients_12/TrResidual_2/truediv_grad/NegNegTrResidual_2/truediv/x*
T0*
_output_shapes
: 
Ł
0gradients_12/TrResidual_2/truediv_grad/RealDiv_1RealDiv*gradients_12/TrResidual_2/truediv_grad/NegTrResidual/alpha/read*
T0*
_output_shapes
:
Š
0gradients_12/TrResidual_2/truediv_grad/RealDiv_2RealDiv0gradients_12/TrResidual_2/truediv_grad/RealDiv_1TrResidual/alpha/read*
T0*
_output_shapes
:
Ú
*gradients_12/TrResidual_2/truediv_grad/mulMulPgradients_12/gradients/TrResidual_2/mul_grad/Mul_1_grad/tuple/control_dependency0gradients_12/TrResidual_2/truediv_grad/RealDiv_2*
T0*
_output_shapes
:
á
,gradients_12/TrResidual_2/truediv_grad/Sum_1Sum*gradients_12/TrResidual_2/truediv_grad/mul>gradients_12/TrResidual_2/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ě
0gradients_12/TrResidual_2/truediv_grad/Reshape_1Reshape,gradients_12/TrResidual_2/truediv_grad/Sum_1.gradients_12/TrResidual_2/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
Ł
7gradients_12/TrResidual_2/truediv_grad/tuple/group_depsNoOp/^gradients_12/TrResidual_2/truediv_grad/Reshape1^gradients_12/TrResidual_2/truediv_grad/Reshape_1

?gradients_12/TrResidual_2/truediv_grad/tuple/control_dependencyIdentity.gradients_12/TrResidual_2/truediv_grad/Reshape8^gradients_12/TrResidual_2/truediv_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_12/TrResidual_2/truediv_grad/Reshape*
_output_shapes
: 
Ł
Agradients_12/TrResidual_2/truediv_grad/tuple/control_dependency_1Identity0gradients_12/TrResidual_2/truediv_grad/Reshape_18^gradients_12/TrResidual_2/truediv_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_12/TrResidual_2/truediv_grad/Reshape_1*
_output_shapes
:

beta1_power_10/initial_valueConst*#
_class
loc:@TrResidual/alpha*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_10
VariableV2*
shared_name *#
_class
loc:@TrResidual/alpha*
	container *
shape: *
dtype0*
_output_shapes
: 
ź
beta1_power_10/AssignAssignbeta1_power_10beta1_power_10/initial_value*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
u
beta1_power_10/readIdentitybeta1_power_10*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
: 

beta2_power_10/initial_valueConst*#
_class
loc:@TrResidual/alpha*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_10
VariableV2*
shared_name *#
_class
loc:@TrResidual/alpha*
	container *
shape: *
dtype0*
_output_shapes
: 
ź
beta2_power_10/AssignAssignbeta2_power_10beta2_power_10/initial_value*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
u
beta2_power_10/readIdentitybeta2_power_10*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
: 

'TrResidual/alpha/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@TrResidual/alpha*
dtype0*
_output_shapes
:
Ś
TrResidual/alpha/Adam
VariableV2*
shared_name *#
_class
loc:@TrResidual/alpha*
	container *
shape:*
dtype0*
_output_shapes
:
Ů
TrResidual/alpha/Adam/AssignAssignTrResidual/alpha/Adam'TrResidual/alpha/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:

TrResidual/alpha/Adam/readIdentityTrResidual/alpha/Adam*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
:

)TrResidual/alpha/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@TrResidual/alpha*
dtype0*
_output_shapes
:
¨
TrResidual/alpha/Adam_1
VariableV2*
shared_name *#
_class
loc:@TrResidual/alpha*
	container *
shape:*
dtype0*
_output_shapes
:
ß
TrResidual/alpha/Adam_1/AssignAssignTrResidual/alpha/Adam_1)TrResidual/alpha/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:

TrResidual/alpha/Adam_1/readIdentityTrResidual/alpha/Adam_1*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
:
Z
Adam_10/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
R
Adam_10/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
R
Adam_10/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
T
Adam_10/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

)Adam_10/update_TrResidual/alpha/ApplyAdam	ApplyAdamTrResidual/alphaTrResidual/alpha/AdamTrResidual/alpha/Adam_1beta1_power_10/readbeta2_power_10/readAdam_10/learning_rateAdam_10/beta1Adam_10/beta2Adam_10/epsilonAgradients_12/TrResidual_2/truediv_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@TrResidual/alpha*
use_nesterov( *
_output_shapes
:
¨
Adam_10/mulMulbeta1_power_10/readAdam_10/beta1*^Adam_10/update_TrResidual/alpha/ApplyAdam*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
: 
¤
Adam_10/AssignAssignbeta1_power_10Adam_10/mul*
use_locking( *
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
Ş
Adam_10/mul_1Mulbeta2_power_10/readAdam_10/beta2*^Adam_10/update_TrResidual/alpha/ApplyAdam*
T0*#
_class
loc:@TrResidual/alpha*
_output_shapes
: 
¨
Adam_10/Assign_1Assignbeta2_power_10Adam_10/mul_1*
use_locking( *
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
_
Adam_10NoOp^Adam_10/Assign^Adam_10/Assign_1*^Adam_10/update_TrResidual/alpha/ApplyAdam
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
ź-
save/SaveV2/tensor_namesConst*î,
valueä,Bá,ÝBClassify/bias1BClassify/bias1/AdamBClassify/bias1/Adam_1BClassify/bias2BClassify/bias2/AdamBClassify/bias2/Adam_1BClassify/bias3BClassify/bias3/AdamBClassify/bias3/Adam_1BClassify/bias4_1BClassify/bias4_1/AdamBClassify/bias4_1/Adam_1BClassify/bias4_2BClassify/bias4_2/AdamBClassify/bias4_2/Adam_1BClassify/bias4_3BClassify/bias4_3/AdamBClassify/bias4_3/Adam_1BClassify/w1BClassify/w1/AdamBClassify/w1/Adam_1BClassify/w2BClassify/w2/AdamBClassify/w2/Adam_1BClassify/w3BClassify/w3/AdamBClassify/w3/Adam_1BClassify/w4_1BClassify/w4_1/AdamBClassify/w4_1/Adam_1BClassify/w4_2BClassify/w4_2/AdamBClassify/w4_2/Adam_1BClassify/w4_3BClassify/w4_3/AdamBClassify/w4_3/Adam_1BRegress/bias1_regBRegress/bias1_reg/AdamBRegress/bias1_reg/Adam_1BRegress/bias1_reg/Adam_2BRegress/bias1_reg/Adam_3BRegress/bias1_reg/Adam_4BRegress/bias1_reg/Adam_5BRegress/bias2_regBRegress/bias2_reg/AdamBRegress/bias2_reg/Adam_1BRegress/bias2_reg/Adam_2BRegress/bias2_reg/Adam_3BRegress/bias2_reg/Adam_4BRegress/bias2_reg/Adam_5BRegress/bias3_regBRegress/bias3_reg/AdamBRegress/bias3_reg/Adam_1BRegress/bias3_reg/Adam_2BRegress/bias3_reg/Adam_3BRegress/bias3_reg/Adam_4BRegress/bias3_reg/Adam_5BRegress/bias4_regBRegress/bias4_reg/AdamBRegress/bias4_reg/Adam_1BRegress/bias4_reg/Adam_2BRegress/bias4_reg/Adam_3BRegress/bias4_reg/Adam_4BRegress/bias4_reg/Adam_5BRegress/w1_regBRegress/w1_reg/AdamBRegress/w1_reg/Adam_1BRegress/w1_reg/Adam_2BRegress/w1_reg/Adam_3BRegress/w1_reg/Adam_4BRegress/w1_reg/Adam_5BRegress/w2_regBRegress/w2_reg/AdamBRegress/w2_reg/Adam_1BRegress/w2_reg/Adam_2BRegress/w2_reg/Adam_3BRegress/w2_reg/Adam_4BRegress/w2_reg/Adam_5BRegress/w3_regBRegress/w3_reg/AdamBRegress/w3_reg/Adam_1BRegress/w3_reg/Adam_2BRegress/w3_reg/Adam_3BRegress/w3_reg/Adam_4BRegress/w3_reg/Adam_5BRegress/w4_regBRegress/w4_reg/AdamBRegress/w4_reg/Adam_1BRegress/w4_reg/Adam_2BRegress/w4_reg/Adam_3BRegress/w4_reg/Adam_4BRegress/w4_reg/Adam_5BResidualRegress/bias1_regBResidualRegress/bias1_reg/AdamB ResidualRegress/bias1_reg/Adam_1B!ResidualRegress/bias1_reg/Adam_10B!ResidualRegress/bias1_reg/Adam_11B ResidualRegress/bias1_reg/Adam_2B ResidualRegress/bias1_reg/Adam_3B ResidualRegress/bias1_reg/Adam_4B ResidualRegress/bias1_reg/Adam_5B ResidualRegress/bias1_reg/Adam_6B ResidualRegress/bias1_reg/Adam_7B ResidualRegress/bias1_reg/Adam_8B ResidualRegress/bias1_reg/Adam_9BResidualRegress/bias2_regBResidualRegress/bias2_reg/AdamB ResidualRegress/bias2_reg/Adam_1B!ResidualRegress/bias2_reg/Adam_10B!ResidualRegress/bias2_reg/Adam_11B ResidualRegress/bias2_reg/Adam_2B ResidualRegress/bias2_reg/Adam_3B ResidualRegress/bias2_reg/Adam_4B ResidualRegress/bias2_reg/Adam_5B ResidualRegress/bias2_reg/Adam_6B ResidualRegress/bias2_reg/Adam_7B ResidualRegress/bias2_reg/Adam_8B ResidualRegress/bias2_reg/Adam_9BResidualRegress/bias3_regBResidualRegress/bias3_reg/AdamB ResidualRegress/bias3_reg/Adam_1B!ResidualRegress/bias3_reg/Adam_10B!ResidualRegress/bias3_reg/Adam_11B ResidualRegress/bias3_reg/Adam_2B ResidualRegress/bias3_reg/Adam_3B ResidualRegress/bias3_reg/Adam_4B ResidualRegress/bias3_reg/Adam_5B ResidualRegress/bias3_reg/Adam_6B ResidualRegress/bias3_reg/Adam_7B ResidualRegress/bias3_reg/Adam_8B ResidualRegress/bias3_reg/Adam_9BResidualRegress/bias4_regBResidualRegress/bias4_reg/AdamB ResidualRegress/bias4_reg/Adam_1B!ResidualRegress/bias4_reg/Adam_10B!ResidualRegress/bias4_reg/Adam_11B ResidualRegress/bias4_reg/Adam_2B ResidualRegress/bias4_reg/Adam_3B ResidualRegress/bias4_reg/Adam_4B ResidualRegress/bias4_reg/Adam_5B ResidualRegress/bias4_reg/Adam_6B ResidualRegress/bias4_reg/Adam_7B ResidualRegress/bias4_reg/Adam_8B ResidualRegress/bias4_reg/Adam_9BResidualRegress/w1_regBResidualRegress/w1_reg/AdamBResidualRegress/w1_reg/Adam_1BResidualRegress/w1_reg/Adam_10BResidualRegress/w1_reg/Adam_11BResidualRegress/w1_reg/Adam_2BResidualRegress/w1_reg/Adam_3BResidualRegress/w1_reg/Adam_4BResidualRegress/w1_reg/Adam_5BResidualRegress/w1_reg/Adam_6BResidualRegress/w1_reg/Adam_7BResidualRegress/w1_reg/Adam_8BResidualRegress/w1_reg/Adam_9BResidualRegress/w2_regBResidualRegress/w2_reg/AdamBResidualRegress/w2_reg/Adam_1BResidualRegress/w2_reg/Adam_10BResidualRegress/w2_reg/Adam_11BResidualRegress/w2_reg/Adam_2BResidualRegress/w2_reg/Adam_3BResidualRegress/w2_reg/Adam_4BResidualRegress/w2_reg/Adam_5BResidualRegress/w2_reg/Adam_6BResidualRegress/w2_reg/Adam_7BResidualRegress/w2_reg/Adam_8BResidualRegress/w2_reg/Adam_9BResidualRegress/w3_regBResidualRegress/w3_reg/AdamBResidualRegress/w3_reg/Adam_1BResidualRegress/w3_reg/Adam_10BResidualRegress/w3_reg/Adam_11BResidualRegress/w3_reg/Adam_2BResidualRegress/w3_reg/Adam_3BResidualRegress/w3_reg/Adam_4BResidualRegress/w3_reg/Adam_5BResidualRegress/w3_reg/Adam_6BResidualRegress/w3_reg/Adam_7BResidualRegress/w3_reg/Adam_8BResidualRegress/w3_reg/Adam_9BResidualRegress/w4_regBResidualRegress/w4_reg/AdamBResidualRegress/w4_reg/Adam_1BResidualRegress/w4_reg/Adam_10BResidualRegress/w4_reg/Adam_11BResidualRegress/w4_reg/Adam_2BResidualRegress/w4_reg/Adam_3BResidualRegress/w4_reg/Adam_4BResidualRegress/w4_reg/Adam_5BResidualRegress/w4_reg/Adam_6BResidualRegress/w4_reg/Adam_7BResidualRegress/w4_reg/Adam_8BResidualRegress/w4_reg/Adam_9BTrResidual/alphaBTrResidual/alpha/AdamBTrResidual/alpha/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_10Bbeta1_power_2Bbeta1_power_3Bbeta1_power_4Bbeta1_power_5Bbeta1_power_6Bbeta1_power_7Bbeta1_power_8Bbeta1_power_9Bbeta2_powerBbeta2_power_1Bbeta2_power_10Bbeta2_power_2Bbeta2_power_3Bbeta2_power_4Bbeta2_power_5Bbeta2_power_6Bbeta2_power_7Bbeta2_power_8Bbeta2_power_9*
dtype0*
_output_shapes	
:Ý
˘
save/SaveV2/shape_and_slicesConst*Đ
valueĆBĂÝB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:Ý
˘/
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesClassify/bias1Classify/bias1/AdamClassify/bias1/Adam_1Classify/bias2Classify/bias2/AdamClassify/bias2/Adam_1Classify/bias3Classify/bias3/AdamClassify/bias3/Adam_1Classify/bias4_1Classify/bias4_1/AdamClassify/bias4_1/Adam_1Classify/bias4_2Classify/bias4_2/AdamClassify/bias4_2/Adam_1Classify/bias4_3Classify/bias4_3/AdamClassify/bias4_3/Adam_1Classify/w1Classify/w1/AdamClassify/w1/Adam_1Classify/w2Classify/w2/AdamClassify/w2/Adam_1Classify/w3Classify/w3/AdamClassify/w3/Adam_1Classify/w4_1Classify/w4_1/AdamClassify/w4_1/Adam_1Classify/w4_2Classify/w4_2/AdamClassify/w4_2/Adam_1Classify/w4_3Classify/w4_3/AdamClassify/w4_3/Adam_1Regress/bias1_regRegress/bias1_reg/AdamRegress/bias1_reg/Adam_1Regress/bias1_reg/Adam_2Regress/bias1_reg/Adam_3Regress/bias1_reg/Adam_4Regress/bias1_reg/Adam_5Regress/bias2_regRegress/bias2_reg/AdamRegress/bias2_reg/Adam_1Regress/bias2_reg/Adam_2Regress/bias2_reg/Adam_3Regress/bias2_reg/Adam_4Regress/bias2_reg/Adam_5Regress/bias3_regRegress/bias3_reg/AdamRegress/bias3_reg/Adam_1Regress/bias3_reg/Adam_2Regress/bias3_reg/Adam_3Regress/bias3_reg/Adam_4Regress/bias3_reg/Adam_5Regress/bias4_regRegress/bias4_reg/AdamRegress/bias4_reg/Adam_1Regress/bias4_reg/Adam_2Regress/bias4_reg/Adam_3Regress/bias4_reg/Adam_4Regress/bias4_reg/Adam_5Regress/w1_regRegress/w1_reg/AdamRegress/w1_reg/Adam_1Regress/w1_reg/Adam_2Regress/w1_reg/Adam_3Regress/w1_reg/Adam_4Regress/w1_reg/Adam_5Regress/w2_regRegress/w2_reg/AdamRegress/w2_reg/Adam_1Regress/w2_reg/Adam_2Regress/w2_reg/Adam_3Regress/w2_reg/Adam_4Regress/w2_reg/Adam_5Regress/w3_regRegress/w3_reg/AdamRegress/w3_reg/Adam_1Regress/w3_reg/Adam_2Regress/w3_reg/Adam_3Regress/w3_reg/Adam_4Regress/w3_reg/Adam_5Regress/w4_regRegress/w4_reg/AdamRegress/w4_reg/Adam_1Regress/w4_reg/Adam_2Regress/w4_reg/Adam_3Regress/w4_reg/Adam_4Regress/w4_reg/Adam_5ResidualRegress/bias1_regResidualRegress/bias1_reg/Adam ResidualRegress/bias1_reg/Adam_1!ResidualRegress/bias1_reg/Adam_10!ResidualRegress/bias1_reg/Adam_11 ResidualRegress/bias1_reg/Adam_2 ResidualRegress/bias1_reg/Adam_3 ResidualRegress/bias1_reg/Adam_4 ResidualRegress/bias1_reg/Adam_5 ResidualRegress/bias1_reg/Adam_6 ResidualRegress/bias1_reg/Adam_7 ResidualRegress/bias1_reg/Adam_8 ResidualRegress/bias1_reg/Adam_9ResidualRegress/bias2_regResidualRegress/bias2_reg/Adam ResidualRegress/bias2_reg/Adam_1!ResidualRegress/bias2_reg/Adam_10!ResidualRegress/bias2_reg/Adam_11 ResidualRegress/bias2_reg/Adam_2 ResidualRegress/bias2_reg/Adam_3 ResidualRegress/bias2_reg/Adam_4 ResidualRegress/bias2_reg/Adam_5 ResidualRegress/bias2_reg/Adam_6 ResidualRegress/bias2_reg/Adam_7 ResidualRegress/bias2_reg/Adam_8 ResidualRegress/bias2_reg/Adam_9ResidualRegress/bias3_regResidualRegress/bias3_reg/Adam ResidualRegress/bias3_reg/Adam_1!ResidualRegress/bias3_reg/Adam_10!ResidualRegress/bias3_reg/Adam_11 ResidualRegress/bias3_reg/Adam_2 ResidualRegress/bias3_reg/Adam_3 ResidualRegress/bias3_reg/Adam_4 ResidualRegress/bias3_reg/Adam_5 ResidualRegress/bias3_reg/Adam_6 ResidualRegress/bias3_reg/Adam_7 ResidualRegress/bias3_reg/Adam_8 ResidualRegress/bias3_reg/Adam_9ResidualRegress/bias4_regResidualRegress/bias4_reg/Adam ResidualRegress/bias4_reg/Adam_1!ResidualRegress/bias4_reg/Adam_10!ResidualRegress/bias4_reg/Adam_11 ResidualRegress/bias4_reg/Adam_2 ResidualRegress/bias4_reg/Adam_3 ResidualRegress/bias4_reg/Adam_4 ResidualRegress/bias4_reg/Adam_5 ResidualRegress/bias4_reg/Adam_6 ResidualRegress/bias4_reg/Adam_7 ResidualRegress/bias4_reg/Adam_8 ResidualRegress/bias4_reg/Adam_9ResidualRegress/w1_regResidualRegress/w1_reg/AdamResidualRegress/w1_reg/Adam_1ResidualRegress/w1_reg/Adam_10ResidualRegress/w1_reg/Adam_11ResidualRegress/w1_reg/Adam_2ResidualRegress/w1_reg/Adam_3ResidualRegress/w1_reg/Adam_4ResidualRegress/w1_reg/Adam_5ResidualRegress/w1_reg/Adam_6ResidualRegress/w1_reg/Adam_7ResidualRegress/w1_reg/Adam_8ResidualRegress/w1_reg/Adam_9ResidualRegress/w2_regResidualRegress/w2_reg/AdamResidualRegress/w2_reg/Adam_1ResidualRegress/w2_reg/Adam_10ResidualRegress/w2_reg/Adam_11ResidualRegress/w2_reg/Adam_2ResidualRegress/w2_reg/Adam_3ResidualRegress/w2_reg/Adam_4ResidualRegress/w2_reg/Adam_5ResidualRegress/w2_reg/Adam_6ResidualRegress/w2_reg/Adam_7ResidualRegress/w2_reg/Adam_8ResidualRegress/w2_reg/Adam_9ResidualRegress/w3_regResidualRegress/w3_reg/AdamResidualRegress/w3_reg/Adam_1ResidualRegress/w3_reg/Adam_10ResidualRegress/w3_reg/Adam_11ResidualRegress/w3_reg/Adam_2ResidualRegress/w3_reg/Adam_3ResidualRegress/w3_reg/Adam_4ResidualRegress/w3_reg/Adam_5ResidualRegress/w3_reg/Adam_6ResidualRegress/w3_reg/Adam_7ResidualRegress/w3_reg/Adam_8ResidualRegress/w3_reg/Adam_9ResidualRegress/w4_regResidualRegress/w4_reg/AdamResidualRegress/w4_reg/Adam_1ResidualRegress/w4_reg/Adam_10ResidualRegress/w4_reg/Adam_11ResidualRegress/w4_reg/Adam_2ResidualRegress/w4_reg/Adam_3ResidualRegress/w4_reg/Adam_4ResidualRegress/w4_reg/Adam_5ResidualRegress/w4_reg/Adam_6ResidualRegress/w4_reg/Adam_7ResidualRegress/w4_reg/Adam_8ResidualRegress/w4_reg/Adam_9TrResidual/alphaTrResidual/alpha/AdamTrResidual/alpha/Adam_1beta1_powerbeta1_power_1beta1_power_10beta1_power_2beta1_power_3beta1_power_4beta1_power_5beta1_power_6beta1_power_7beta1_power_8beta1_power_9beta2_powerbeta2_power_1beta2_power_10beta2_power_2beta2_power_3beta2_power_4beta2_power_5beta2_power_6beta2_power_7beta2_power_8beta2_power_9*î
dtypesă
ŕ2Ý
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Î-
save/RestoreV2/tensor_namesConst"/device:CPU:0*î,
valueä,Bá,ÝBClassify/bias1BClassify/bias1/AdamBClassify/bias1/Adam_1BClassify/bias2BClassify/bias2/AdamBClassify/bias2/Adam_1BClassify/bias3BClassify/bias3/AdamBClassify/bias3/Adam_1BClassify/bias4_1BClassify/bias4_1/AdamBClassify/bias4_1/Adam_1BClassify/bias4_2BClassify/bias4_2/AdamBClassify/bias4_2/Adam_1BClassify/bias4_3BClassify/bias4_3/AdamBClassify/bias4_3/Adam_1BClassify/w1BClassify/w1/AdamBClassify/w1/Adam_1BClassify/w2BClassify/w2/AdamBClassify/w2/Adam_1BClassify/w3BClassify/w3/AdamBClassify/w3/Adam_1BClassify/w4_1BClassify/w4_1/AdamBClassify/w4_1/Adam_1BClassify/w4_2BClassify/w4_2/AdamBClassify/w4_2/Adam_1BClassify/w4_3BClassify/w4_3/AdamBClassify/w4_3/Adam_1BRegress/bias1_regBRegress/bias1_reg/AdamBRegress/bias1_reg/Adam_1BRegress/bias1_reg/Adam_2BRegress/bias1_reg/Adam_3BRegress/bias1_reg/Adam_4BRegress/bias1_reg/Adam_5BRegress/bias2_regBRegress/bias2_reg/AdamBRegress/bias2_reg/Adam_1BRegress/bias2_reg/Adam_2BRegress/bias2_reg/Adam_3BRegress/bias2_reg/Adam_4BRegress/bias2_reg/Adam_5BRegress/bias3_regBRegress/bias3_reg/AdamBRegress/bias3_reg/Adam_1BRegress/bias3_reg/Adam_2BRegress/bias3_reg/Adam_3BRegress/bias3_reg/Adam_4BRegress/bias3_reg/Adam_5BRegress/bias4_regBRegress/bias4_reg/AdamBRegress/bias4_reg/Adam_1BRegress/bias4_reg/Adam_2BRegress/bias4_reg/Adam_3BRegress/bias4_reg/Adam_4BRegress/bias4_reg/Adam_5BRegress/w1_regBRegress/w1_reg/AdamBRegress/w1_reg/Adam_1BRegress/w1_reg/Adam_2BRegress/w1_reg/Adam_3BRegress/w1_reg/Adam_4BRegress/w1_reg/Adam_5BRegress/w2_regBRegress/w2_reg/AdamBRegress/w2_reg/Adam_1BRegress/w2_reg/Adam_2BRegress/w2_reg/Adam_3BRegress/w2_reg/Adam_4BRegress/w2_reg/Adam_5BRegress/w3_regBRegress/w3_reg/AdamBRegress/w3_reg/Adam_1BRegress/w3_reg/Adam_2BRegress/w3_reg/Adam_3BRegress/w3_reg/Adam_4BRegress/w3_reg/Adam_5BRegress/w4_regBRegress/w4_reg/AdamBRegress/w4_reg/Adam_1BRegress/w4_reg/Adam_2BRegress/w4_reg/Adam_3BRegress/w4_reg/Adam_4BRegress/w4_reg/Adam_5BResidualRegress/bias1_regBResidualRegress/bias1_reg/AdamB ResidualRegress/bias1_reg/Adam_1B!ResidualRegress/bias1_reg/Adam_10B!ResidualRegress/bias1_reg/Adam_11B ResidualRegress/bias1_reg/Adam_2B ResidualRegress/bias1_reg/Adam_3B ResidualRegress/bias1_reg/Adam_4B ResidualRegress/bias1_reg/Adam_5B ResidualRegress/bias1_reg/Adam_6B ResidualRegress/bias1_reg/Adam_7B ResidualRegress/bias1_reg/Adam_8B ResidualRegress/bias1_reg/Adam_9BResidualRegress/bias2_regBResidualRegress/bias2_reg/AdamB ResidualRegress/bias2_reg/Adam_1B!ResidualRegress/bias2_reg/Adam_10B!ResidualRegress/bias2_reg/Adam_11B ResidualRegress/bias2_reg/Adam_2B ResidualRegress/bias2_reg/Adam_3B ResidualRegress/bias2_reg/Adam_4B ResidualRegress/bias2_reg/Adam_5B ResidualRegress/bias2_reg/Adam_6B ResidualRegress/bias2_reg/Adam_7B ResidualRegress/bias2_reg/Adam_8B ResidualRegress/bias2_reg/Adam_9BResidualRegress/bias3_regBResidualRegress/bias3_reg/AdamB ResidualRegress/bias3_reg/Adam_1B!ResidualRegress/bias3_reg/Adam_10B!ResidualRegress/bias3_reg/Adam_11B ResidualRegress/bias3_reg/Adam_2B ResidualRegress/bias3_reg/Adam_3B ResidualRegress/bias3_reg/Adam_4B ResidualRegress/bias3_reg/Adam_5B ResidualRegress/bias3_reg/Adam_6B ResidualRegress/bias3_reg/Adam_7B ResidualRegress/bias3_reg/Adam_8B ResidualRegress/bias3_reg/Adam_9BResidualRegress/bias4_regBResidualRegress/bias4_reg/AdamB ResidualRegress/bias4_reg/Adam_1B!ResidualRegress/bias4_reg/Adam_10B!ResidualRegress/bias4_reg/Adam_11B ResidualRegress/bias4_reg/Adam_2B ResidualRegress/bias4_reg/Adam_3B ResidualRegress/bias4_reg/Adam_4B ResidualRegress/bias4_reg/Adam_5B ResidualRegress/bias4_reg/Adam_6B ResidualRegress/bias4_reg/Adam_7B ResidualRegress/bias4_reg/Adam_8B ResidualRegress/bias4_reg/Adam_9BResidualRegress/w1_regBResidualRegress/w1_reg/AdamBResidualRegress/w1_reg/Adam_1BResidualRegress/w1_reg/Adam_10BResidualRegress/w1_reg/Adam_11BResidualRegress/w1_reg/Adam_2BResidualRegress/w1_reg/Adam_3BResidualRegress/w1_reg/Adam_4BResidualRegress/w1_reg/Adam_5BResidualRegress/w1_reg/Adam_6BResidualRegress/w1_reg/Adam_7BResidualRegress/w1_reg/Adam_8BResidualRegress/w1_reg/Adam_9BResidualRegress/w2_regBResidualRegress/w2_reg/AdamBResidualRegress/w2_reg/Adam_1BResidualRegress/w2_reg/Adam_10BResidualRegress/w2_reg/Adam_11BResidualRegress/w2_reg/Adam_2BResidualRegress/w2_reg/Adam_3BResidualRegress/w2_reg/Adam_4BResidualRegress/w2_reg/Adam_5BResidualRegress/w2_reg/Adam_6BResidualRegress/w2_reg/Adam_7BResidualRegress/w2_reg/Adam_8BResidualRegress/w2_reg/Adam_9BResidualRegress/w3_regBResidualRegress/w3_reg/AdamBResidualRegress/w3_reg/Adam_1BResidualRegress/w3_reg/Adam_10BResidualRegress/w3_reg/Adam_11BResidualRegress/w3_reg/Adam_2BResidualRegress/w3_reg/Adam_3BResidualRegress/w3_reg/Adam_4BResidualRegress/w3_reg/Adam_5BResidualRegress/w3_reg/Adam_6BResidualRegress/w3_reg/Adam_7BResidualRegress/w3_reg/Adam_8BResidualRegress/w3_reg/Adam_9BResidualRegress/w4_regBResidualRegress/w4_reg/AdamBResidualRegress/w4_reg/Adam_1BResidualRegress/w4_reg/Adam_10BResidualRegress/w4_reg/Adam_11BResidualRegress/w4_reg/Adam_2BResidualRegress/w4_reg/Adam_3BResidualRegress/w4_reg/Adam_4BResidualRegress/w4_reg/Adam_5BResidualRegress/w4_reg/Adam_6BResidualRegress/w4_reg/Adam_7BResidualRegress/w4_reg/Adam_8BResidualRegress/w4_reg/Adam_9BTrResidual/alphaBTrResidual/alpha/AdamBTrResidual/alpha/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_10Bbeta1_power_2Bbeta1_power_3Bbeta1_power_4Bbeta1_power_5Bbeta1_power_6Bbeta1_power_7Bbeta1_power_8Bbeta1_power_9Bbeta2_powerBbeta2_power_1Bbeta2_power_10Bbeta2_power_2Bbeta2_power_3Bbeta2_power_4Bbeta2_power_5Bbeta2_power_6Bbeta2_power_7Bbeta2_power_8Bbeta2_power_9*
dtype0*
_output_shapes	
:Ý
´
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*Đ
valueĆBĂÝB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:Ý
ň	
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*î
dtypesă
ŕ2Ý*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
§
save/AssignAssignClassify/bias1save/RestoreV2*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:
°
save/Assign_1AssignClassify/bias1/Adamsave/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:
˛
save/Assign_2AssignClassify/bias1/Adam_1save/RestoreV2:2*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes	
:
Ť
save/Assign_3AssignClassify/bias2save/RestoreV2:3*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:
°
save/Assign_4AssignClassify/bias2/Adamsave/RestoreV2:4*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:
˛
save/Assign_5AssignClassify/bias2/Adam_1save/RestoreV2:5*
use_locking(*
T0*!
_class
loc:@Classify/bias2*
validate_shape(*
_output_shapes	
:
Ť
save/Assign_6AssignClassify/bias3save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:
°
save/Assign_7AssignClassify/bias3/Adamsave/RestoreV2:7*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:
˛
save/Assign_8AssignClassify/bias3/Adam_1save/RestoreV2:8*
use_locking(*
T0*!
_class
loc:@Classify/bias3*
validate_shape(*
_output_shapes	
:
Ž
save/Assign_9AssignClassify/bias4_1save/RestoreV2:9*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:
ľ
save/Assign_10AssignClassify/bias4_1/Adamsave/RestoreV2:10*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_11AssignClassify/bias4_1/Adam_1save/RestoreV2:11*
use_locking(*
T0*#
_class
loc:@Classify/bias4_1*
validate_shape(*
_output_shapes
:
°
save/Assign_12AssignClassify/bias4_2save/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:
ľ
save/Assign_13AssignClassify/bias4_2/Adamsave/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_14AssignClassify/bias4_2/Adam_1save/RestoreV2:14*
use_locking(*
T0*#
_class
loc:@Classify/bias4_2*
validate_shape(*
_output_shapes
:
°
save/Assign_15AssignClassify/bias4_3save/RestoreV2:15*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:
ľ
save/Assign_16AssignClassify/bias4_3/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_17AssignClassify/bias4_3/Adam_1save/RestoreV2:17*
use_locking(*
T0*#
_class
loc:@Classify/bias4_3*
validate_shape(*
_output_shapes
:
Ť
save/Assign_18AssignClassify/w1save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2
°
save/Assign_19AssignClassify/w1/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2
˛
save/Assign_20AssignClassify/w1/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Classify/w1*
validate_shape(*
_output_shapes
:	2
Ź
save/Assign_21AssignClassify/w2save/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:

ą
save/Assign_22AssignClassify/w2/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:

ł
save/Assign_23AssignClassify/w2/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Classify/w2*
validate_shape(* 
_output_shapes
:

Ź
save/Assign_24AssignClassify/w3save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:

ą
save/Assign_25AssignClassify/w3/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:

ł
save/Assign_26AssignClassify/w3/Adam_1save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Classify/w3*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_27AssignClassify/w4_1save/RestoreV2:27*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	
´
save/Assign_28AssignClassify/w4_1/Adamsave/RestoreV2:28*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	
ś
save/Assign_29AssignClassify/w4_1/Adam_1save/RestoreV2:29*
use_locking(*
T0* 
_class
loc:@Classify/w4_1*
validate_shape(*
_output_shapes
:	
Ż
save/Assign_30AssignClassify/w4_2save/RestoreV2:30*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	
´
save/Assign_31AssignClassify/w4_2/Adamsave/RestoreV2:31*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	
ś
save/Assign_32AssignClassify/w4_2/Adam_1save/RestoreV2:32*
use_locking(*
T0* 
_class
loc:@Classify/w4_2*
validate_shape(*
_output_shapes
:	
Ż
save/Assign_33AssignClassify/w4_3save/RestoreV2:33*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	
´
save/Assign_34AssignClassify/w4_3/Adamsave/RestoreV2:34*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	
ś
save/Assign_35AssignClassify/w4_3/Adam_1save/RestoreV2:35*
use_locking(*
T0* 
_class
loc:@Classify/w4_3*
validate_shape(*
_output_shapes
:	
ł
save/Assign_36AssignRegress/bias1_regsave/RestoreV2:36*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
¸
save/Assign_37AssignRegress/bias1_reg/Adamsave/RestoreV2:37*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_38AssignRegress/bias1_reg/Adam_1save/RestoreV2:38*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_39AssignRegress/bias1_reg/Adam_2save/RestoreV2:39*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_40AssignRegress/bias1_reg/Adam_3save/RestoreV2:40*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_41AssignRegress/bias1_reg/Adam_4save/RestoreV2:41*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_42AssignRegress/bias1_reg/Adam_5save/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes	
:
ł
save/Assign_43AssignRegress/bias2_regsave/RestoreV2:43*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
¸
save/Assign_44AssignRegress/bias2_reg/Adamsave/RestoreV2:44*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_45AssignRegress/bias2_reg/Adam_1save/RestoreV2:45*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_46AssignRegress/bias2_reg/Adam_2save/RestoreV2:46*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_47AssignRegress/bias2_reg/Adam_3save/RestoreV2:47*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_48AssignRegress/bias2_reg/Adam_4save/RestoreV2:48*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_49AssignRegress/bias2_reg/Adam_5save/RestoreV2:49*
use_locking(*
T0*$
_class
loc:@Regress/bias2_reg*
validate_shape(*
_output_shapes	
:
ł
save/Assign_50AssignRegress/bias3_regsave/RestoreV2:50*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
¸
save/Assign_51AssignRegress/bias3_reg/Adamsave/RestoreV2:51*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_52AssignRegress/bias3_reg/Adam_1save/RestoreV2:52*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_53AssignRegress/bias3_reg/Adam_2save/RestoreV2:53*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_54AssignRegress/bias3_reg/Adam_3save/RestoreV2:54*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_55AssignRegress/bias3_reg/Adam_4save/RestoreV2:55*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
ş
save/Assign_56AssignRegress/bias3_reg/Adam_5save/RestoreV2:56*
use_locking(*
T0*$
_class
loc:@Regress/bias3_reg*
validate_shape(*
_output_shapes	
:
˛
save/Assign_57AssignRegress/bias4_regsave/RestoreV2:57*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_58AssignRegress/bias4_reg/Adamsave/RestoreV2:58*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
š
save/Assign_59AssignRegress/bias4_reg/Adam_1save/RestoreV2:59*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
š
save/Assign_60AssignRegress/bias4_reg/Adam_2save/RestoreV2:60*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
š
save/Assign_61AssignRegress/bias4_reg/Adam_3save/RestoreV2:61*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
š
save/Assign_62AssignRegress/bias4_reg/Adam_4save/RestoreV2:62*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
š
save/Assign_63AssignRegress/bias4_reg/Adam_5save/RestoreV2:63*
use_locking(*
T0*$
_class
loc:@Regress/bias4_reg*
validate_shape(*
_output_shapes
:
ą
save/Assign_64AssignRegress/w1_regsave/RestoreV2:64*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
ś
save/Assign_65AssignRegress/w1_reg/Adamsave/RestoreV2:65*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
¸
save/Assign_66AssignRegress/w1_reg/Adam_1save/RestoreV2:66*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
¸
save/Assign_67AssignRegress/w1_reg/Adam_2save/RestoreV2:67*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
¸
save/Assign_68AssignRegress/w1_reg/Adam_3save/RestoreV2:68*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
¸
save/Assign_69AssignRegress/w1_reg/Adam_4save/RestoreV2:69*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
¸
save/Assign_70AssignRegress/w1_reg/Adam_5save/RestoreV2:70*
use_locking(*
T0*!
_class
loc:@Regress/w1_reg*
validate_shape(*
_output_shapes
:	2
˛
save/Assign_71AssignRegress/w2_regsave/RestoreV2:71*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

ˇ
save/Assign_72AssignRegress/w2_reg/Adamsave/RestoreV2:72*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_73AssignRegress/w2_reg/Adam_1save/RestoreV2:73*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_74AssignRegress/w2_reg/Adam_2save/RestoreV2:74*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_75AssignRegress/w2_reg/Adam_3save/RestoreV2:75*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_76AssignRegress/w2_reg/Adam_4save/RestoreV2:76*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_77AssignRegress/w2_reg/Adam_5save/RestoreV2:77*
use_locking(*
T0*!
_class
loc:@Regress/w2_reg*
validate_shape(* 
_output_shapes
:

˛
save/Assign_78AssignRegress/w3_regsave/RestoreV2:78*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

ˇ
save/Assign_79AssignRegress/w3_reg/Adamsave/RestoreV2:79*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_80AssignRegress/w3_reg/Adam_1save/RestoreV2:80*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_81AssignRegress/w3_reg/Adam_2save/RestoreV2:81*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_82AssignRegress/w3_reg/Adam_3save/RestoreV2:82*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_83AssignRegress/w3_reg/Adam_4save/RestoreV2:83*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

š
save/Assign_84AssignRegress/w3_reg/Adam_5save/RestoreV2:84*
use_locking(*
T0*!
_class
loc:@Regress/w3_reg*
validate_shape(* 
_output_shapes
:

ą
save/Assign_85AssignRegress/w4_regsave/RestoreV2:85*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
ś
save/Assign_86AssignRegress/w4_reg/Adamsave/RestoreV2:86*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
¸
save/Assign_87AssignRegress/w4_reg/Adam_1save/RestoreV2:87*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
¸
save/Assign_88AssignRegress/w4_reg/Adam_2save/RestoreV2:88*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
¸
save/Assign_89AssignRegress/w4_reg/Adam_3save/RestoreV2:89*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
¸
save/Assign_90AssignRegress/w4_reg/Adam_4save/RestoreV2:90*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
¸
save/Assign_91AssignRegress/w4_reg/Adam_5save/RestoreV2:91*
use_locking(*
T0*!
_class
loc:@Regress/w4_reg*
validate_shape(*
_output_shapes
:	
Ă
save/Assign_92AssignResidualRegress/bias1_regsave/RestoreV2:92*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Č
save/Assign_93AssignResidualRegress/bias1_reg/Adamsave/RestoreV2:93*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_94Assign ResidualRegress/bias1_reg/Adam_1save/RestoreV2:94*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ë
save/Assign_95Assign!ResidualRegress/bias1_reg/Adam_10save/RestoreV2:95*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ë
save/Assign_96Assign!ResidualRegress/bias1_reg/Adam_11save/RestoreV2:96*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_97Assign ResidualRegress/bias1_reg/Adam_2save/RestoreV2:97*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_98Assign ResidualRegress/bias1_reg/Adam_3save/RestoreV2:98*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_99Assign ResidualRegress/bias1_reg/Adam_4save/RestoreV2:99*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_100Assign ResidualRegress/bias1_reg/Adam_5save/RestoreV2:100*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_101Assign ResidualRegress/bias1_reg/Adam_6save/RestoreV2:101*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_102Assign ResidualRegress/bias1_reg/Adam_7save/RestoreV2:102*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_103Assign ResidualRegress/bias1_reg/Adam_8save/RestoreV2:103*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_104Assign ResidualRegress/bias1_reg/Adam_9save/RestoreV2:104*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes	
:
Ĺ
save/Assign_105AssignResidualRegress/bias2_regsave/RestoreV2:105*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_106AssignResidualRegress/bias2_reg/Adamsave/RestoreV2:106*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_107Assign ResidualRegress/bias2_reg/Adam_1save/RestoreV2:107*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Í
save/Assign_108Assign!ResidualRegress/bias2_reg/Adam_10save/RestoreV2:108*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Í
save/Assign_109Assign!ResidualRegress/bias2_reg/Adam_11save/RestoreV2:109*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_110Assign ResidualRegress/bias2_reg/Adam_2save/RestoreV2:110*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_111Assign ResidualRegress/bias2_reg/Adam_3save/RestoreV2:111*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_112Assign ResidualRegress/bias2_reg/Adam_4save/RestoreV2:112*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_113Assign ResidualRegress/bias2_reg/Adam_5save/RestoreV2:113*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_114Assign ResidualRegress/bias2_reg/Adam_6save/RestoreV2:114*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_115Assign ResidualRegress/bias2_reg/Adam_7save/RestoreV2:115*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_116Assign ResidualRegress/bias2_reg/Adam_8save/RestoreV2:116*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_117Assign ResidualRegress/bias2_reg/Adam_9save/RestoreV2:117*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias2_reg*
validate_shape(*
_output_shapes	
:
Ĺ
save/Assign_118AssignResidualRegress/bias3_regsave/RestoreV2:118*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ę
save/Assign_119AssignResidualRegress/bias3_reg/Adamsave/RestoreV2:119*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_120Assign ResidualRegress/bias3_reg/Adam_1save/RestoreV2:120*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Í
save/Assign_121Assign!ResidualRegress/bias3_reg/Adam_10save/RestoreV2:121*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Í
save/Assign_122Assign!ResidualRegress/bias3_reg/Adam_11save/RestoreV2:122*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_123Assign ResidualRegress/bias3_reg/Adam_2save/RestoreV2:123*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_124Assign ResidualRegress/bias3_reg/Adam_3save/RestoreV2:124*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_125Assign ResidualRegress/bias3_reg/Adam_4save/RestoreV2:125*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_126Assign ResidualRegress/bias3_reg/Adam_5save/RestoreV2:126*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_127Assign ResidualRegress/bias3_reg/Adam_6save/RestoreV2:127*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_128Assign ResidualRegress/bias3_reg/Adam_7save/RestoreV2:128*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_129Assign ResidualRegress/bias3_reg/Adam_8save/RestoreV2:129*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ě
save/Assign_130Assign ResidualRegress/bias3_reg/Adam_9save/RestoreV2:130*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias3_reg*
validate_shape(*
_output_shapes	
:
Ä
save/Assign_131AssignResidualRegress/bias4_regsave/RestoreV2:131*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
É
save/Assign_132AssignResidualRegress/bias4_reg/Adamsave/RestoreV2:132*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_133Assign ResidualRegress/bias4_reg/Adam_1save/RestoreV2:133*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ě
save/Assign_134Assign!ResidualRegress/bias4_reg/Adam_10save/RestoreV2:134*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ě
save/Assign_135Assign!ResidualRegress/bias4_reg/Adam_11save/RestoreV2:135*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_136Assign ResidualRegress/bias4_reg/Adam_2save/RestoreV2:136*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_137Assign ResidualRegress/bias4_reg/Adam_3save/RestoreV2:137*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_138Assign ResidualRegress/bias4_reg/Adam_4save/RestoreV2:138*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_139Assign ResidualRegress/bias4_reg/Adam_5save/RestoreV2:139*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_140Assign ResidualRegress/bias4_reg/Adam_6save/RestoreV2:140*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_141Assign ResidualRegress/bias4_reg/Adam_7save/RestoreV2:141*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_142Assign ResidualRegress/bias4_reg/Adam_8save/RestoreV2:142*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ë
save/Assign_143Assign ResidualRegress/bias4_reg/Adam_9save/RestoreV2:143*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias4_reg*
validate_shape(*
_output_shapes
:
Ă
save/Assign_144AssignResidualRegress/w1_regsave/RestoreV2:144*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Č
save/Assign_145AssignResidualRegress/w1_reg/Adamsave/RestoreV2:145*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_146AssignResidualRegress/w1_reg/Adam_1save/RestoreV2:146*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ë
save/Assign_147AssignResidualRegress/w1_reg/Adam_10save/RestoreV2:147*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ë
save/Assign_148AssignResidualRegress/w1_reg/Adam_11save/RestoreV2:148*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_149AssignResidualRegress/w1_reg/Adam_2save/RestoreV2:149*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_150AssignResidualRegress/w1_reg/Adam_3save/RestoreV2:150*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_151AssignResidualRegress/w1_reg/Adam_4save/RestoreV2:151*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_152AssignResidualRegress/w1_reg/Adam_5save/RestoreV2:152*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_153AssignResidualRegress/w1_reg/Adam_6save/RestoreV2:153*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_154AssignResidualRegress/w1_reg/Adam_7save/RestoreV2:154*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_155AssignResidualRegress/w1_reg/Adam_8save/RestoreV2:155*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ę
save/Assign_156AssignResidualRegress/w1_reg/Adam_9save/RestoreV2:156*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w1_reg*
validate_shape(*
_output_shapes
:	5
Ä
save/Assign_157AssignResidualRegress/w2_regsave/RestoreV2:157*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

É
save/Assign_158AssignResidualRegress/w2_reg/Adamsave/RestoreV2:158*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_159AssignResidualRegress/w2_reg/Adam_1save/RestoreV2:159*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ě
save/Assign_160AssignResidualRegress/w2_reg/Adam_10save/RestoreV2:160*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ě
save/Assign_161AssignResidualRegress/w2_reg/Adam_11save/RestoreV2:161*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_162AssignResidualRegress/w2_reg/Adam_2save/RestoreV2:162*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_163AssignResidualRegress/w2_reg/Adam_3save/RestoreV2:163*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_164AssignResidualRegress/w2_reg/Adam_4save/RestoreV2:164*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_165AssignResidualRegress/w2_reg/Adam_5save/RestoreV2:165*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_166AssignResidualRegress/w2_reg/Adam_6save/RestoreV2:166*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_167AssignResidualRegress/w2_reg/Adam_7save/RestoreV2:167*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_168AssignResidualRegress/w2_reg/Adam_8save/RestoreV2:168*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_169AssignResidualRegress/w2_reg/Adam_9save/RestoreV2:169*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w2_reg*
validate_shape(* 
_output_shapes
:

Ä
save/Assign_170AssignResidualRegress/w3_regsave/RestoreV2:170*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

É
save/Assign_171AssignResidualRegress/w3_reg/Adamsave/RestoreV2:171*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_172AssignResidualRegress/w3_reg/Adam_1save/RestoreV2:172*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ě
save/Assign_173AssignResidualRegress/w3_reg/Adam_10save/RestoreV2:173*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ě
save/Assign_174AssignResidualRegress/w3_reg/Adam_11save/RestoreV2:174*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_175AssignResidualRegress/w3_reg/Adam_2save/RestoreV2:175*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_176AssignResidualRegress/w3_reg/Adam_3save/RestoreV2:176*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_177AssignResidualRegress/w3_reg/Adam_4save/RestoreV2:177*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_178AssignResidualRegress/w3_reg/Adam_5save/RestoreV2:178*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_179AssignResidualRegress/w3_reg/Adam_6save/RestoreV2:179*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_180AssignResidualRegress/w3_reg/Adam_7save/RestoreV2:180*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_181AssignResidualRegress/w3_reg/Adam_8save/RestoreV2:181*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ë
save/Assign_182AssignResidualRegress/w3_reg/Adam_9save/RestoreV2:182*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w3_reg*
validate_shape(* 
_output_shapes
:

Ă
save/Assign_183AssignResidualRegress/w4_regsave/RestoreV2:183*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Č
save/Assign_184AssignResidualRegress/w4_reg/Adamsave/RestoreV2:184*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_185AssignResidualRegress/w4_reg/Adam_1save/RestoreV2:185*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ë
save/Assign_186AssignResidualRegress/w4_reg/Adam_10save/RestoreV2:186*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ë
save/Assign_187AssignResidualRegress/w4_reg/Adam_11save/RestoreV2:187*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_188AssignResidualRegress/w4_reg/Adam_2save/RestoreV2:188*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_189AssignResidualRegress/w4_reg/Adam_3save/RestoreV2:189*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_190AssignResidualRegress/w4_reg/Adam_4save/RestoreV2:190*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_191AssignResidualRegress/w4_reg/Adam_5save/RestoreV2:191*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_192AssignResidualRegress/w4_reg/Adam_6save/RestoreV2:192*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_193AssignResidualRegress/w4_reg/Adam_7save/RestoreV2:193*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_194AssignResidualRegress/w4_reg/Adam_8save/RestoreV2:194*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
Ę
save/Assign_195AssignResidualRegress/w4_reg/Adam_9save/RestoreV2:195*
use_locking(*
T0*)
_class
loc:@ResidualRegress/w4_reg*
validate_shape(*
_output_shapes
:	
˛
save/Assign_196AssignTrResidual/alphasave/RestoreV2:196*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:
ˇ
save/Assign_197AssignTrResidual/alpha/Adamsave/RestoreV2:197*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:
š
save/Assign_198AssignTrResidual/alpha/Adam_1save/RestoreV2:198*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
:
§
save/Assign_199Assignbeta1_powersave/RestoreV2:199*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_200Assignbeta1_power_1save/RestoreV2:200*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_201Assignbeta1_power_10save/RestoreV2:201*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_202Assignbeta1_power_2save/RestoreV2:202*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_203Assignbeta1_power_3save/RestoreV2:203*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_204Assignbeta1_power_4save/RestoreV2:204*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_205Assignbeta1_power_5save/RestoreV2:205*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_206Assignbeta1_power_6save/RestoreV2:206*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_207Assignbeta1_power_7save/RestoreV2:207*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_208Assignbeta1_power_8save/RestoreV2:208*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_209Assignbeta1_power_9save/RestoreV2:209*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
§
save/Assign_210Assignbeta2_powersave/RestoreV2:210*
use_locking(*
T0*!
_class
loc:@Classify/bias1*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_211Assignbeta2_power_1save/RestoreV2:211*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_212Assignbeta2_power_10save/RestoreV2:212*
use_locking(*
T0*#
_class
loc:@TrResidual/alpha*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_213Assignbeta2_power_2save/RestoreV2:213*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
Ź
save/Assign_214Assignbeta2_power_3save/RestoreV2:214*
use_locking(*
T0*$
_class
loc:@Regress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_215Assignbeta2_power_4save/RestoreV2:215*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_216Assignbeta2_power_5save/RestoreV2:216*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_217Assignbeta2_power_6save/RestoreV2:217*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_218Assignbeta2_power_7save/RestoreV2:218*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_219Assignbeta2_power_8save/RestoreV2:219*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
´
save/Assign_220Assignbeta2_power_9save/RestoreV2:220*
use_locking(*
T0*,
_class"
 loc:@ResidualRegress/bias1_reg*
validate_shape(*
_output_shapes
: 
˛
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_167^save/Assign_168^save/Assign_169^save/Assign_17^save/Assign_170^save/Assign_171^save/Assign_172^save/Assign_173^save/Assign_174^save/Assign_175^save/Assign_176^save/Assign_177^save/Assign_178^save/Assign_179^save/Assign_18^save/Assign_180^save/Assign_181^save/Assign_182^save/Assign_183^save/Assign_184^save/Assign_185^save/Assign_186^save/Assign_187^save/Assign_188^save/Assign_189^save/Assign_19^save/Assign_190^save/Assign_191^save/Assign_192^save/Assign_193^save/Assign_194^save/Assign_195^save/Assign_196^save/Assign_197^save/Assign_198^save/Assign_199^save/Assign_2^save/Assign_20^save/Assign_200^save/Assign_201^save/Assign_202^save/Assign_203^save/Assign_204^save/Assign_205^save/Assign_206^save/Assign_207^save/Assign_208^save/Assign_209^save/Assign_21^save/Assign_210^save/Assign_211^save/Assign_212^save/Assign_213^save/Assign_214^save/Assign_215^save/Assign_216^save/Assign_217^save/Assign_218^save/Assign_219^save/Assign_22^save/Assign_220^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
Ě:
initNoOp^Classify/bias1/Adam/Assign^Classify/bias1/Adam_1/Assign^Classify/bias1/Assign^Classify/bias2/Adam/Assign^Classify/bias2/Adam_1/Assign^Classify/bias2/Assign^Classify/bias3/Adam/Assign^Classify/bias3/Adam_1/Assign^Classify/bias3/Assign^Classify/bias4_1/Adam/Assign^Classify/bias4_1/Adam_1/Assign^Classify/bias4_1/Assign^Classify/bias4_2/Adam/Assign^Classify/bias4_2/Adam_1/Assign^Classify/bias4_2/Assign^Classify/bias4_3/Adam/Assign^Classify/bias4_3/Adam_1/Assign^Classify/bias4_3/Assign^Classify/w1/Adam/Assign^Classify/w1/Adam_1/Assign^Classify/w1/Assign^Classify/w2/Adam/Assign^Classify/w2/Adam_1/Assign^Classify/w2/Assign^Classify/w3/Adam/Assign^Classify/w3/Adam_1/Assign^Classify/w3/Assign^Classify/w4_1/Adam/Assign^Classify/w4_1/Adam_1/Assign^Classify/w4_1/Assign^Classify/w4_2/Adam/Assign^Classify/w4_2/Adam_1/Assign^Classify/w4_2/Assign^Classify/w4_3/Adam/Assign^Classify/w4_3/Adam_1/Assign^Classify/w4_3/Assign^Regress/bias1_reg/Adam/Assign ^Regress/bias1_reg/Adam_1/Assign ^Regress/bias1_reg/Adam_2/Assign ^Regress/bias1_reg/Adam_3/Assign ^Regress/bias1_reg/Adam_4/Assign ^Regress/bias1_reg/Adam_5/Assign^Regress/bias1_reg/Assign^Regress/bias2_reg/Adam/Assign ^Regress/bias2_reg/Adam_1/Assign ^Regress/bias2_reg/Adam_2/Assign ^Regress/bias2_reg/Adam_3/Assign ^Regress/bias2_reg/Adam_4/Assign ^Regress/bias2_reg/Adam_5/Assign^Regress/bias2_reg/Assign^Regress/bias3_reg/Adam/Assign ^Regress/bias3_reg/Adam_1/Assign ^Regress/bias3_reg/Adam_2/Assign ^Regress/bias3_reg/Adam_3/Assign ^Regress/bias3_reg/Adam_4/Assign ^Regress/bias3_reg/Adam_5/Assign^Regress/bias3_reg/Assign^Regress/bias4_reg/Adam/Assign ^Regress/bias4_reg/Adam_1/Assign ^Regress/bias4_reg/Adam_2/Assign ^Regress/bias4_reg/Adam_3/Assign ^Regress/bias4_reg/Adam_4/Assign ^Regress/bias4_reg/Adam_5/Assign^Regress/bias4_reg/Assign^Regress/w1_reg/Adam/Assign^Regress/w1_reg/Adam_1/Assign^Regress/w1_reg/Adam_2/Assign^Regress/w1_reg/Adam_3/Assign^Regress/w1_reg/Adam_4/Assign^Regress/w1_reg/Adam_5/Assign^Regress/w1_reg/Assign^Regress/w2_reg/Adam/Assign^Regress/w2_reg/Adam_1/Assign^Regress/w2_reg/Adam_2/Assign^Regress/w2_reg/Adam_3/Assign^Regress/w2_reg/Adam_4/Assign^Regress/w2_reg/Adam_5/Assign^Regress/w2_reg/Assign^Regress/w3_reg/Adam/Assign^Regress/w3_reg/Adam_1/Assign^Regress/w3_reg/Adam_2/Assign^Regress/w3_reg/Adam_3/Assign^Regress/w3_reg/Adam_4/Assign^Regress/w3_reg/Adam_5/Assign^Regress/w3_reg/Assign^Regress/w4_reg/Adam/Assign^Regress/w4_reg/Adam_1/Assign^Regress/w4_reg/Adam_2/Assign^Regress/w4_reg/Adam_3/Assign^Regress/w4_reg/Adam_4/Assign^Regress/w4_reg/Adam_5/Assign^Regress/w4_reg/Assign&^ResidualRegress/bias1_reg/Adam/Assign(^ResidualRegress/bias1_reg/Adam_1/Assign)^ResidualRegress/bias1_reg/Adam_10/Assign)^ResidualRegress/bias1_reg/Adam_11/Assign(^ResidualRegress/bias1_reg/Adam_2/Assign(^ResidualRegress/bias1_reg/Adam_3/Assign(^ResidualRegress/bias1_reg/Adam_4/Assign(^ResidualRegress/bias1_reg/Adam_5/Assign(^ResidualRegress/bias1_reg/Adam_6/Assign(^ResidualRegress/bias1_reg/Adam_7/Assign(^ResidualRegress/bias1_reg/Adam_8/Assign(^ResidualRegress/bias1_reg/Adam_9/Assign!^ResidualRegress/bias1_reg/Assign&^ResidualRegress/bias2_reg/Adam/Assign(^ResidualRegress/bias2_reg/Adam_1/Assign)^ResidualRegress/bias2_reg/Adam_10/Assign)^ResidualRegress/bias2_reg/Adam_11/Assign(^ResidualRegress/bias2_reg/Adam_2/Assign(^ResidualRegress/bias2_reg/Adam_3/Assign(^ResidualRegress/bias2_reg/Adam_4/Assign(^ResidualRegress/bias2_reg/Adam_5/Assign(^ResidualRegress/bias2_reg/Adam_6/Assign(^ResidualRegress/bias2_reg/Adam_7/Assign(^ResidualRegress/bias2_reg/Adam_8/Assign(^ResidualRegress/bias2_reg/Adam_9/Assign!^ResidualRegress/bias2_reg/Assign&^ResidualRegress/bias3_reg/Adam/Assign(^ResidualRegress/bias3_reg/Adam_1/Assign)^ResidualRegress/bias3_reg/Adam_10/Assign)^ResidualRegress/bias3_reg/Adam_11/Assign(^ResidualRegress/bias3_reg/Adam_2/Assign(^ResidualRegress/bias3_reg/Adam_3/Assign(^ResidualRegress/bias3_reg/Adam_4/Assign(^ResidualRegress/bias3_reg/Adam_5/Assign(^ResidualRegress/bias3_reg/Adam_6/Assign(^ResidualRegress/bias3_reg/Adam_7/Assign(^ResidualRegress/bias3_reg/Adam_8/Assign(^ResidualRegress/bias3_reg/Adam_9/Assign!^ResidualRegress/bias3_reg/Assign&^ResidualRegress/bias4_reg/Adam/Assign(^ResidualRegress/bias4_reg/Adam_1/Assign)^ResidualRegress/bias4_reg/Adam_10/Assign)^ResidualRegress/bias4_reg/Adam_11/Assign(^ResidualRegress/bias4_reg/Adam_2/Assign(^ResidualRegress/bias4_reg/Adam_3/Assign(^ResidualRegress/bias4_reg/Adam_4/Assign(^ResidualRegress/bias4_reg/Adam_5/Assign(^ResidualRegress/bias4_reg/Adam_6/Assign(^ResidualRegress/bias4_reg/Adam_7/Assign(^ResidualRegress/bias4_reg/Adam_8/Assign(^ResidualRegress/bias4_reg/Adam_9/Assign!^ResidualRegress/bias4_reg/Assign#^ResidualRegress/w1_reg/Adam/Assign%^ResidualRegress/w1_reg/Adam_1/Assign&^ResidualRegress/w1_reg/Adam_10/Assign&^ResidualRegress/w1_reg/Adam_11/Assign%^ResidualRegress/w1_reg/Adam_2/Assign%^ResidualRegress/w1_reg/Adam_3/Assign%^ResidualRegress/w1_reg/Adam_4/Assign%^ResidualRegress/w1_reg/Adam_5/Assign%^ResidualRegress/w1_reg/Adam_6/Assign%^ResidualRegress/w1_reg/Adam_7/Assign%^ResidualRegress/w1_reg/Adam_8/Assign%^ResidualRegress/w1_reg/Adam_9/Assign^ResidualRegress/w1_reg/Assign#^ResidualRegress/w2_reg/Adam/Assign%^ResidualRegress/w2_reg/Adam_1/Assign&^ResidualRegress/w2_reg/Adam_10/Assign&^ResidualRegress/w2_reg/Adam_11/Assign%^ResidualRegress/w2_reg/Adam_2/Assign%^ResidualRegress/w2_reg/Adam_3/Assign%^ResidualRegress/w2_reg/Adam_4/Assign%^ResidualRegress/w2_reg/Adam_5/Assign%^ResidualRegress/w2_reg/Adam_6/Assign%^ResidualRegress/w2_reg/Adam_7/Assign%^ResidualRegress/w2_reg/Adam_8/Assign%^ResidualRegress/w2_reg/Adam_9/Assign^ResidualRegress/w2_reg/Assign#^ResidualRegress/w3_reg/Adam/Assign%^ResidualRegress/w3_reg/Adam_1/Assign&^ResidualRegress/w3_reg/Adam_10/Assign&^ResidualRegress/w3_reg/Adam_11/Assign%^ResidualRegress/w3_reg/Adam_2/Assign%^ResidualRegress/w3_reg/Adam_3/Assign%^ResidualRegress/w3_reg/Adam_4/Assign%^ResidualRegress/w3_reg/Adam_5/Assign%^ResidualRegress/w3_reg/Adam_6/Assign%^ResidualRegress/w3_reg/Adam_7/Assign%^ResidualRegress/w3_reg/Adam_8/Assign%^ResidualRegress/w3_reg/Adam_9/Assign^ResidualRegress/w3_reg/Assign#^ResidualRegress/w4_reg/Adam/Assign%^ResidualRegress/w4_reg/Adam_1/Assign&^ResidualRegress/w4_reg/Adam_10/Assign&^ResidualRegress/w4_reg/Adam_11/Assign%^ResidualRegress/w4_reg/Adam_2/Assign%^ResidualRegress/w4_reg/Adam_3/Assign%^ResidualRegress/w4_reg/Adam_4/Assign%^ResidualRegress/w4_reg/Adam_5/Assign%^ResidualRegress/w4_reg/Adam_6/Assign%^ResidualRegress/w4_reg/Adam_7/Assign%^ResidualRegress/w4_reg/Adam_8/Assign%^ResidualRegress/w4_reg/Adam_9/Assign^ResidualRegress/w4_reg/Assign^TrResidual/alpha/Adam/Assign^TrResidual/alpha/Adam_1/Assign^TrResidual/alpha/Assign^beta1_power/Assign^beta1_power_1/Assign^beta1_power_10/Assign^beta1_power_2/Assign^beta1_power_3/Assign^beta1_power_4/Assign^beta1_power_5/Assign^beta1_power_6/Assign^beta1_power_7/Assign^beta1_power_8/Assign^beta1_power_9/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_10/Assign^beta2_power_2/Assign^beta2_power_3/Assign^beta2_power_4/Assign^beta2_power_5/Assign^beta2_power_6/Assign^beta2_power_7/Assign^beta2_power_8/Assign^beta2_power_9/Assign"B
save/Const:0save/control_dependency:0save/restore_all5 @F8"đ
lossesĺ
â
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0
$softmax_cross_entropy_loss_2/value:0
$softmax_cross_entropy_loss_3/value:0
$softmax_cross_entropy_loss_4/value:0
$softmax_cross_entropy_loss_5/value:0"ł
trainable_variables
b
Classify/w1:0Classify/w1/AssignClassify/w1/read:02'Classify/w1/Initializer/random_normal:08
f
Classify/bias1:0Classify/bias1/AssignClassify/bias1/read:02"Classify/bias1/Initializer/Const:08
b
Classify/w2:0Classify/w2/AssignClassify/w2/read:02'Classify/w2/Initializer/random_normal:08
f
Classify/bias2:0Classify/bias2/AssignClassify/bias2/read:02"Classify/bias2/Initializer/Const:08
b
Classify/w3:0Classify/w3/AssignClassify/w3/read:02'Classify/w3/Initializer/random_normal:08
f
Classify/bias3:0Classify/bias3/AssignClassify/bias3/read:02"Classify/bias3/Initializer/Const:08
j
Classify/w4_1:0Classify/w4_1/AssignClassify/w4_1/read:02)Classify/w4_1/Initializer/random_normal:08
n
Classify/bias4_1:0Classify/bias4_1/AssignClassify/bias4_1/read:02$Classify/bias4_1/Initializer/Const:08
j
Classify/w4_2:0Classify/w4_2/AssignClassify/w4_2/read:02)Classify/w4_2/Initializer/random_normal:08
n
Classify/bias4_2:0Classify/bias4_2/AssignClassify/bias4_2/read:02$Classify/bias4_2/Initializer/Const:08
j
Classify/w4_3:0Classify/w4_3/AssignClassify/w4_3/read:02)Classify/w4_3/Initializer/random_normal:08
n
Classify/bias4_3:0Classify/bias4_3/AssignClassify/bias4_3/read:02$Classify/bias4_3/Initializer/Const:08

ResidualRegress/w1_reg:0ResidualRegress/w1_reg/AssignResidualRegress/w1_reg/read:022ResidualRegress/w1_reg/Initializer/random_normal:08

ResidualRegress/bias1_reg:0 ResidualRegress/bias1_reg/Assign ResidualRegress/bias1_reg/read:02-ResidualRegress/bias1_reg/Initializer/Const:08

ResidualRegress/w2_reg:0ResidualRegress/w2_reg/AssignResidualRegress/w2_reg/read:022ResidualRegress/w2_reg/Initializer/random_normal:08

ResidualRegress/bias2_reg:0 ResidualRegress/bias2_reg/Assign ResidualRegress/bias2_reg/read:02-ResidualRegress/bias2_reg/Initializer/Const:08

ResidualRegress/w3_reg:0ResidualRegress/w3_reg/AssignResidualRegress/w3_reg/read:022ResidualRegress/w3_reg/Initializer/random_normal:08

ResidualRegress/bias3_reg:0 ResidualRegress/bias3_reg/Assign ResidualRegress/bias3_reg/read:02-ResidualRegress/bias3_reg/Initializer/Const:08

ResidualRegress/w4_reg:0ResidualRegress/w4_reg/AssignResidualRegress/w4_reg/read:022ResidualRegress/w4_reg/Initializer/random_normal:08

ResidualRegress/bias4_reg:0 ResidualRegress/bias4_reg/Assign ResidualRegress/bias4_reg/read:02-ResidualRegress/bias4_reg/Initializer/Const:08
n
Regress/w1_reg:0Regress/w1_reg/AssignRegress/w1_reg/read:02*Regress/w1_reg/Initializer/random_normal:08
r
Regress/bias1_reg:0Regress/bias1_reg/AssignRegress/bias1_reg/read:02%Regress/bias1_reg/Initializer/Const:08
n
Regress/w2_reg:0Regress/w2_reg/AssignRegress/w2_reg/read:02*Regress/w2_reg/Initializer/random_normal:08
r
Regress/bias2_reg:0Regress/bias2_reg/AssignRegress/bias2_reg/read:02%Regress/bias2_reg/Initializer/Const:08
n
Regress/w3_reg:0Regress/w3_reg/AssignRegress/w3_reg/read:02*Regress/w3_reg/Initializer/random_normal:08
r
Regress/bias3_reg:0Regress/bias3_reg/AssignRegress/bias3_reg/read:02%Regress/bias3_reg/Initializer/Const:08
n
Regress/w4_reg:0Regress/w4_reg/AssignRegress/w4_reg/read:02*Regress/w4_reg/Initializer/random_normal:08
r
Regress/bias4_reg:0Regress/bias4_reg/AssignRegress/bias4_reg/read:02%Regress/bias4_reg/Initializer/Const:08
v
TrResidual/alpha:0TrResidual/alpha/AssignTrResidual/alpha/read:02,TrResidual/alpha/Initializer/random_normal:08"e
train_opY
W
Adam
Adam_1
Adam_2
Adam_3
Adam_4
Adam_5
Adam_6
Adam_7
Adam_8
Adam_9
Adam_10"şö
	variablesŤö§ö
b
Classify/w1:0Classify/w1/AssignClassify/w1/read:02'Classify/w1/Initializer/random_normal:08
f
Classify/bias1:0Classify/bias1/AssignClassify/bias1/read:02"Classify/bias1/Initializer/Const:08
b
Classify/w2:0Classify/w2/AssignClassify/w2/read:02'Classify/w2/Initializer/random_normal:08
f
Classify/bias2:0Classify/bias2/AssignClassify/bias2/read:02"Classify/bias2/Initializer/Const:08
b
Classify/w3:0Classify/w3/AssignClassify/w3/read:02'Classify/w3/Initializer/random_normal:08
f
Classify/bias3:0Classify/bias3/AssignClassify/bias3/read:02"Classify/bias3/Initializer/Const:08
j
Classify/w4_1:0Classify/w4_1/AssignClassify/w4_1/read:02)Classify/w4_1/Initializer/random_normal:08
n
Classify/bias4_1:0Classify/bias4_1/AssignClassify/bias4_1/read:02$Classify/bias4_1/Initializer/Const:08
j
Classify/w4_2:0Classify/w4_2/AssignClassify/w4_2/read:02)Classify/w4_2/Initializer/random_normal:08
n
Classify/bias4_2:0Classify/bias4_2/AssignClassify/bias4_2/read:02$Classify/bias4_2/Initializer/Const:08
j
Classify/w4_3:0Classify/w4_3/AssignClassify/w4_3/read:02)Classify/w4_3/Initializer/random_normal:08
n
Classify/bias4_3:0Classify/bias4_3/AssignClassify/bias4_3/read:02$Classify/bias4_3/Initializer/Const:08

ResidualRegress/w1_reg:0ResidualRegress/w1_reg/AssignResidualRegress/w1_reg/read:022ResidualRegress/w1_reg/Initializer/random_normal:08

ResidualRegress/bias1_reg:0 ResidualRegress/bias1_reg/Assign ResidualRegress/bias1_reg/read:02-ResidualRegress/bias1_reg/Initializer/Const:08

ResidualRegress/w2_reg:0ResidualRegress/w2_reg/AssignResidualRegress/w2_reg/read:022ResidualRegress/w2_reg/Initializer/random_normal:08

ResidualRegress/bias2_reg:0 ResidualRegress/bias2_reg/Assign ResidualRegress/bias2_reg/read:02-ResidualRegress/bias2_reg/Initializer/Const:08

ResidualRegress/w3_reg:0ResidualRegress/w3_reg/AssignResidualRegress/w3_reg/read:022ResidualRegress/w3_reg/Initializer/random_normal:08

ResidualRegress/bias3_reg:0 ResidualRegress/bias3_reg/Assign ResidualRegress/bias3_reg/read:02-ResidualRegress/bias3_reg/Initializer/Const:08

ResidualRegress/w4_reg:0ResidualRegress/w4_reg/AssignResidualRegress/w4_reg/read:022ResidualRegress/w4_reg/Initializer/random_normal:08

ResidualRegress/bias4_reg:0 ResidualRegress/bias4_reg/Assign ResidualRegress/bias4_reg/read:02-ResidualRegress/bias4_reg/Initializer/Const:08
n
Regress/w1_reg:0Regress/w1_reg/AssignRegress/w1_reg/read:02*Regress/w1_reg/Initializer/random_normal:08
r
Regress/bias1_reg:0Regress/bias1_reg/AssignRegress/bias1_reg/read:02%Regress/bias1_reg/Initializer/Const:08
n
Regress/w2_reg:0Regress/w2_reg/AssignRegress/w2_reg/read:02*Regress/w2_reg/Initializer/random_normal:08
r
Regress/bias2_reg:0Regress/bias2_reg/AssignRegress/bias2_reg/read:02%Regress/bias2_reg/Initializer/Const:08
n
Regress/w3_reg:0Regress/w3_reg/AssignRegress/w3_reg/read:02*Regress/w3_reg/Initializer/random_normal:08
r
Regress/bias3_reg:0Regress/bias3_reg/AssignRegress/bias3_reg/read:02%Regress/bias3_reg/Initializer/Const:08
n
Regress/w4_reg:0Regress/w4_reg/AssignRegress/w4_reg/read:02*Regress/w4_reg/Initializer/random_normal:08
r
Regress/bias4_reg:0Regress/bias4_reg/AssignRegress/bias4_reg/read:02%Regress/bias4_reg/Initializer/Const:08
v
TrResidual/alpha:0TrResidual/alpha/AssignTrResidual/alpha/read:02,TrResidual/alpha/Initializer/random_normal:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
l
Classify/w1/Adam:0Classify/w1/Adam/AssignClassify/w1/Adam/read:02$Classify/w1/Adam/Initializer/zeros:0
t
Classify/w1/Adam_1:0Classify/w1/Adam_1/AssignClassify/w1/Adam_1/read:02&Classify/w1/Adam_1/Initializer/zeros:0
x
Classify/bias1/Adam:0Classify/bias1/Adam/AssignClassify/bias1/Adam/read:02'Classify/bias1/Adam/Initializer/zeros:0

Classify/bias1/Adam_1:0Classify/bias1/Adam_1/AssignClassify/bias1/Adam_1/read:02)Classify/bias1/Adam_1/Initializer/zeros:0
l
Classify/w2/Adam:0Classify/w2/Adam/AssignClassify/w2/Adam/read:02$Classify/w2/Adam/Initializer/zeros:0
t
Classify/w2/Adam_1:0Classify/w2/Adam_1/AssignClassify/w2/Adam_1/read:02&Classify/w2/Adam_1/Initializer/zeros:0
x
Classify/bias2/Adam:0Classify/bias2/Adam/AssignClassify/bias2/Adam/read:02'Classify/bias2/Adam/Initializer/zeros:0

Classify/bias2/Adam_1:0Classify/bias2/Adam_1/AssignClassify/bias2/Adam_1/read:02)Classify/bias2/Adam_1/Initializer/zeros:0
l
Classify/w3/Adam:0Classify/w3/Adam/AssignClassify/w3/Adam/read:02$Classify/w3/Adam/Initializer/zeros:0
t
Classify/w3/Adam_1:0Classify/w3/Adam_1/AssignClassify/w3/Adam_1/read:02&Classify/w3/Adam_1/Initializer/zeros:0
x
Classify/bias3/Adam:0Classify/bias3/Adam/AssignClassify/bias3/Adam/read:02'Classify/bias3/Adam/Initializer/zeros:0

Classify/bias3/Adam_1:0Classify/bias3/Adam_1/AssignClassify/bias3/Adam_1/read:02)Classify/bias3/Adam_1/Initializer/zeros:0
t
Classify/w4_1/Adam:0Classify/w4_1/Adam/AssignClassify/w4_1/Adam/read:02&Classify/w4_1/Adam/Initializer/zeros:0
|
Classify/w4_1/Adam_1:0Classify/w4_1/Adam_1/AssignClassify/w4_1/Adam_1/read:02(Classify/w4_1/Adam_1/Initializer/zeros:0

Classify/bias4_1/Adam:0Classify/bias4_1/Adam/AssignClassify/bias4_1/Adam/read:02)Classify/bias4_1/Adam/Initializer/zeros:0

Classify/bias4_1/Adam_1:0Classify/bias4_1/Adam_1/AssignClassify/bias4_1/Adam_1/read:02+Classify/bias4_1/Adam_1/Initializer/zeros:0
t
Classify/w4_2/Adam:0Classify/w4_2/Adam/AssignClassify/w4_2/Adam/read:02&Classify/w4_2/Adam/Initializer/zeros:0
|
Classify/w4_2/Adam_1:0Classify/w4_2/Adam_1/AssignClassify/w4_2/Adam_1/read:02(Classify/w4_2/Adam_1/Initializer/zeros:0

Classify/bias4_2/Adam:0Classify/bias4_2/Adam/AssignClassify/bias4_2/Adam/read:02)Classify/bias4_2/Adam/Initializer/zeros:0

Classify/bias4_2/Adam_1:0Classify/bias4_2/Adam_1/AssignClassify/bias4_2/Adam_1/read:02+Classify/bias4_2/Adam_1/Initializer/zeros:0
t
Classify/w4_3/Adam:0Classify/w4_3/Adam/AssignClassify/w4_3/Adam/read:02&Classify/w4_3/Adam/Initializer/zeros:0
|
Classify/w4_3/Adam_1:0Classify/w4_3/Adam_1/AssignClassify/w4_3/Adam_1/read:02(Classify/w4_3/Adam_1/Initializer/zeros:0

Classify/bias4_3/Adam:0Classify/bias4_3/Adam/AssignClassify/bias4_3/Adam/read:02)Classify/bias4_3/Adam/Initializer/zeros:0

Classify/bias4_3/Adam_1:0Classify/bias4_3/Adam_1/AssignClassify/bias4_3/Adam_1/read:02+Classify/bias4_3/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
Regress/w1_reg/Adam:0Regress/w1_reg/Adam/AssignRegress/w1_reg/Adam/read:02'Regress/w1_reg/Adam/Initializer/zeros:0

Regress/w1_reg/Adam_1:0Regress/w1_reg/Adam_1/AssignRegress/w1_reg/Adam_1/read:02)Regress/w1_reg/Adam_1/Initializer/zeros:0

Regress/bias1_reg/Adam:0Regress/bias1_reg/Adam/AssignRegress/bias1_reg/Adam/read:02*Regress/bias1_reg/Adam/Initializer/zeros:0

Regress/bias1_reg/Adam_1:0Regress/bias1_reg/Adam_1/AssignRegress/bias1_reg/Adam_1/read:02,Regress/bias1_reg/Adam_1/Initializer/zeros:0
x
Regress/w2_reg/Adam:0Regress/w2_reg/Adam/AssignRegress/w2_reg/Adam/read:02'Regress/w2_reg/Adam/Initializer/zeros:0

Regress/w2_reg/Adam_1:0Regress/w2_reg/Adam_1/AssignRegress/w2_reg/Adam_1/read:02)Regress/w2_reg/Adam_1/Initializer/zeros:0

Regress/bias2_reg/Adam:0Regress/bias2_reg/Adam/AssignRegress/bias2_reg/Adam/read:02*Regress/bias2_reg/Adam/Initializer/zeros:0

Regress/bias2_reg/Adam_1:0Regress/bias2_reg/Adam_1/AssignRegress/bias2_reg/Adam_1/read:02,Regress/bias2_reg/Adam_1/Initializer/zeros:0
x
Regress/w3_reg/Adam:0Regress/w3_reg/Adam/AssignRegress/w3_reg/Adam/read:02'Regress/w3_reg/Adam/Initializer/zeros:0

Regress/w3_reg/Adam_1:0Regress/w3_reg/Adam_1/AssignRegress/w3_reg/Adam_1/read:02)Regress/w3_reg/Adam_1/Initializer/zeros:0

Regress/bias3_reg/Adam:0Regress/bias3_reg/Adam/AssignRegress/bias3_reg/Adam/read:02*Regress/bias3_reg/Adam/Initializer/zeros:0

Regress/bias3_reg/Adam_1:0Regress/bias3_reg/Adam_1/AssignRegress/bias3_reg/Adam_1/read:02,Regress/bias3_reg/Adam_1/Initializer/zeros:0
x
Regress/w4_reg/Adam:0Regress/w4_reg/Adam/AssignRegress/w4_reg/Adam/read:02'Regress/w4_reg/Adam/Initializer/zeros:0

Regress/w4_reg/Adam_1:0Regress/w4_reg/Adam_1/AssignRegress/w4_reg/Adam_1/read:02)Regress/w4_reg/Adam_1/Initializer/zeros:0

Regress/bias4_reg/Adam:0Regress/bias4_reg/Adam/AssignRegress/bias4_reg/Adam/read:02*Regress/bias4_reg/Adam/Initializer/zeros:0

Regress/bias4_reg/Adam_1:0Regress/bias4_reg/Adam_1/AssignRegress/bias4_reg/Adam_1/read:02,Regress/bias4_reg/Adam_1/Initializer/zeros:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0

Regress/w1_reg/Adam_2:0Regress/w1_reg/Adam_2/AssignRegress/w1_reg/Adam_2/read:02)Regress/w1_reg/Adam_2/Initializer/zeros:0

Regress/w1_reg/Adam_3:0Regress/w1_reg/Adam_3/AssignRegress/w1_reg/Adam_3/read:02)Regress/w1_reg/Adam_3/Initializer/zeros:0

Regress/bias1_reg/Adam_2:0Regress/bias1_reg/Adam_2/AssignRegress/bias1_reg/Adam_2/read:02,Regress/bias1_reg/Adam_2/Initializer/zeros:0

Regress/bias1_reg/Adam_3:0Regress/bias1_reg/Adam_3/AssignRegress/bias1_reg/Adam_3/read:02,Regress/bias1_reg/Adam_3/Initializer/zeros:0

Regress/w2_reg/Adam_2:0Regress/w2_reg/Adam_2/AssignRegress/w2_reg/Adam_2/read:02)Regress/w2_reg/Adam_2/Initializer/zeros:0

Regress/w2_reg/Adam_3:0Regress/w2_reg/Adam_3/AssignRegress/w2_reg/Adam_3/read:02)Regress/w2_reg/Adam_3/Initializer/zeros:0

Regress/bias2_reg/Adam_2:0Regress/bias2_reg/Adam_2/AssignRegress/bias2_reg/Adam_2/read:02,Regress/bias2_reg/Adam_2/Initializer/zeros:0

Regress/bias2_reg/Adam_3:0Regress/bias2_reg/Adam_3/AssignRegress/bias2_reg/Adam_3/read:02,Regress/bias2_reg/Adam_3/Initializer/zeros:0

Regress/w3_reg/Adam_2:0Regress/w3_reg/Adam_2/AssignRegress/w3_reg/Adam_2/read:02)Regress/w3_reg/Adam_2/Initializer/zeros:0

Regress/w3_reg/Adam_3:0Regress/w3_reg/Adam_3/AssignRegress/w3_reg/Adam_3/read:02)Regress/w3_reg/Adam_3/Initializer/zeros:0

Regress/bias3_reg/Adam_2:0Regress/bias3_reg/Adam_2/AssignRegress/bias3_reg/Adam_2/read:02,Regress/bias3_reg/Adam_2/Initializer/zeros:0

Regress/bias3_reg/Adam_3:0Regress/bias3_reg/Adam_3/AssignRegress/bias3_reg/Adam_3/read:02,Regress/bias3_reg/Adam_3/Initializer/zeros:0

Regress/w4_reg/Adam_2:0Regress/w4_reg/Adam_2/AssignRegress/w4_reg/Adam_2/read:02)Regress/w4_reg/Adam_2/Initializer/zeros:0

Regress/w4_reg/Adam_3:0Regress/w4_reg/Adam_3/AssignRegress/w4_reg/Adam_3/read:02)Regress/w4_reg/Adam_3/Initializer/zeros:0

Regress/bias4_reg/Adam_2:0Regress/bias4_reg/Adam_2/AssignRegress/bias4_reg/Adam_2/read:02,Regress/bias4_reg/Adam_2/Initializer/zeros:0

Regress/bias4_reg/Adam_3:0Regress/bias4_reg/Adam_3/AssignRegress/bias4_reg/Adam_3/read:02,Regress/bias4_reg/Adam_3/Initializer/zeros:0
\
beta1_power_3:0beta1_power_3/Assignbeta1_power_3/read:02beta1_power_3/initial_value:0
\
beta2_power_3:0beta2_power_3/Assignbeta2_power_3/read:02beta2_power_3/initial_value:0

Regress/w1_reg/Adam_4:0Regress/w1_reg/Adam_4/AssignRegress/w1_reg/Adam_4/read:02)Regress/w1_reg/Adam_4/Initializer/zeros:0

Regress/w1_reg/Adam_5:0Regress/w1_reg/Adam_5/AssignRegress/w1_reg/Adam_5/read:02)Regress/w1_reg/Adam_5/Initializer/zeros:0

Regress/bias1_reg/Adam_4:0Regress/bias1_reg/Adam_4/AssignRegress/bias1_reg/Adam_4/read:02,Regress/bias1_reg/Adam_4/Initializer/zeros:0

Regress/bias1_reg/Adam_5:0Regress/bias1_reg/Adam_5/AssignRegress/bias1_reg/Adam_5/read:02,Regress/bias1_reg/Adam_5/Initializer/zeros:0

Regress/w2_reg/Adam_4:0Regress/w2_reg/Adam_4/AssignRegress/w2_reg/Adam_4/read:02)Regress/w2_reg/Adam_4/Initializer/zeros:0

Regress/w2_reg/Adam_5:0Regress/w2_reg/Adam_5/AssignRegress/w2_reg/Adam_5/read:02)Regress/w2_reg/Adam_5/Initializer/zeros:0

Regress/bias2_reg/Adam_4:0Regress/bias2_reg/Adam_4/AssignRegress/bias2_reg/Adam_4/read:02,Regress/bias2_reg/Adam_4/Initializer/zeros:0

Regress/bias2_reg/Adam_5:0Regress/bias2_reg/Adam_5/AssignRegress/bias2_reg/Adam_5/read:02,Regress/bias2_reg/Adam_5/Initializer/zeros:0

Regress/w3_reg/Adam_4:0Regress/w3_reg/Adam_4/AssignRegress/w3_reg/Adam_4/read:02)Regress/w3_reg/Adam_4/Initializer/zeros:0

Regress/w3_reg/Adam_5:0Regress/w3_reg/Adam_5/AssignRegress/w3_reg/Adam_5/read:02)Regress/w3_reg/Adam_5/Initializer/zeros:0

Regress/bias3_reg/Adam_4:0Regress/bias3_reg/Adam_4/AssignRegress/bias3_reg/Adam_4/read:02,Regress/bias3_reg/Adam_4/Initializer/zeros:0

Regress/bias3_reg/Adam_5:0Regress/bias3_reg/Adam_5/AssignRegress/bias3_reg/Adam_5/read:02,Regress/bias3_reg/Adam_5/Initializer/zeros:0

Regress/w4_reg/Adam_4:0Regress/w4_reg/Adam_4/AssignRegress/w4_reg/Adam_4/read:02)Regress/w4_reg/Adam_4/Initializer/zeros:0

Regress/w4_reg/Adam_5:0Regress/w4_reg/Adam_5/AssignRegress/w4_reg/Adam_5/read:02)Regress/w4_reg/Adam_5/Initializer/zeros:0

Regress/bias4_reg/Adam_4:0Regress/bias4_reg/Adam_4/AssignRegress/bias4_reg/Adam_4/read:02,Regress/bias4_reg/Adam_4/Initializer/zeros:0

Regress/bias4_reg/Adam_5:0Regress/bias4_reg/Adam_5/AssignRegress/bias4_reg/Adam_5/read:02,Regress/bias4_reg/Adam_5/Initializer/zeros:0
\
beta1_power_4:0beta1_power_4/Assignbeta1_power_4/read:02beta1_power_4/initial_value:0
\
beta2_power_4:0beta2_power_4/Assignbeta2_power_4/read:02beta2_power_4/initial_value:0

ResidualRegress/w1_reg/Adam:0"ResidualRegress/w1_reg/Adam/Assign"ResidualRegress/w1_reg/Adam/read:02/ResidualRegress/w1_reg/Adam/Initializer/zeros:0
 
ResidualRegress/w1_reg/Adam_1:0$ResidualRegress/w1_reg/Adam_1/Assign$ResidualRegress/w1_reg/Adam_1/read:021ResidualRegress/w1_reg/Adam_1/Initializer/zeros:0
¤
 ResidualRegress/bias1_reg/Adam:0%ResidualRegress/bias1_reg/Adam/Assign%ResidualRegress/bias1_reg/Adam/read:022ResidualRegress/bias1_reg/Adam/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_1:0'ResidualRegress/bias1_reg/Adam_1/Assign'ResidualRegress/bias1_reg/Adam_1/read:024ResidualRegress/bias1_reg/Adam_1/Initializer/zeros:0

ResidualRegress/w2_reg/Adam:0"ResidualRegress/w2_reg/Adam/Assign"ResidualRegress/w2_reg/Adam/read:02/ResidualRegress/w2_reg/Adam/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_1:0$ResidualRegress/w2_reg/Adam_1/Assign$ResidualRegress/w2_reg/Adam_1/read:021ResidualRegress/w2_reg/Adam_1/Initializer/zeros:0
¤
 ResidualRegress/bias2_reg/Adam:0%ResidualRegress/bias2_reg/Adam/Assign%ResidualRegress/bias2_reg/Adam/read:022ResidualRegress/bias2_reg/Adam/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_1:0'ResidualRegress/bias2_reg/Adam_1/Assign'ResidualRegress/bias2_reg/Adam_1/read:024ResidualRegress/bias2_reg/Adam_1/Initializer/zeros:0

ResidualRegress/w3_reg/Adam:0"ResidualRegress/w3_reg/Adam/Assign"ResidualRegress/w3_reg/Adam/read:02/ResidualRegress/w3_reg/Adam/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_1:0$ResidualRegress/w3_reg/Adam_1/Assign$ResidualRegress/w3_reg/Adam_1/read:021ResidualRegress/w3_reg/Adam_1/Initializer/zeros:0
¤
 ResidualRegress/bias3_reg/Adam:0%ResidualRegress/bias3_reg/Adam/Assign%ResidualRegress/bias3_reg/Adam/read:022ResidualRegress/bias3_reg/Adam/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_1:0'ResidualRegress/bias3_reg/Adam_1/Assign'ResidualRegress/bias3_reg/Adam_1/read:024ResidualRegress/bias3_reg/Adam_1/Initializer/zeros:0

ResidualRegress/w4_reg/Adam:0"ResidualRegress/w4_reg/Adam/Assign"ResidualRegress/w4_reg/Adam/read:02/ResidualRegress/w4_reg/Adam/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_1:0$ResidualRegress/w4_reg/Adam_1/Assign$ResidualRegress/w4_reg/Adam_1/read:021ResidualRegress/w4_reg/Adam_1/Initializer/zeros:0
¤
 ResidualRegress/bias4_reg/Adam:0%ResidualRegress/bias4_reg/Adam/Assign%ResidualRegress/bias4_reg/Adam/read:022ResidualRegress/bias4_reg/Adam/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_1:0'ResidualRegress/bias4_reg/Adam_1/Assign'ResidualRegress/bias4_reg/Adam_1/read:024ResidualRegress/bias4_reg/Adam_1/Initializer/zeros:0
\
beta1_power_5:0beta1_power_5/Assignbeta1_power_5/read:02beta1_power_5/initial_value:0
\
beta2_power_5:0beta2_power_5/Assignbeta2_power_5/read:02beta2_power_5/initial_value:0
 
ResidualRegress/w1_reg/Adam_2:0$ResidualRegress/w1_reg/Adam_2/Assign$ResidualRegress/w1_reg/Adam_2/read:021ResidualRegress/w1_reg/Adam_2/Initializer/zeros:0
 
ResidualRegress/w1_reg/Adam_3:0$ResidualRegress/w1_reg/Adam_3/Assign$ResidualRegress/w1_reg/Adam_3/read:021ResidualRegress/w1_reg/Adam_3/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_2:0'ResidualRegress/bias1_reg/Adam_2/Assign'ResidualRegress/bias1_reg/Adam_2/read:024ResidualRegress/bias1_reg/Adam_2/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_3:0'ResidualRegress/bias1_reg/Adam_3/Assign'ResidualRegress/bias1_reg/Adam_3/read:024ResidualRegress/bias1_reg/Adam_3/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_2:0$ResidualRegress/w2_reg/Adam_2/Assign$ResidualRegress/w2_reg/Adam_2/read:021ResidualRegress/w2_reg/Adam_2/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_3:0$ResidualRegress/w2_reg/Adam_3/Assign$ResidualRegress/w2_reg/Adam_3/read:021ResidualRegress/w2_reg/Adam_3/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_2:0'ResidualRegress/bias2_reg/Adam_2/Assign'ResidualRegress/bias2_reg/Adam_2/read:024ResidualRegress/bias2_reg/Adam_2/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_3:0'ResidualRegress/bias2_reg/Adam_3/Assign'ResidualRegress/bias2_reg/Adam_3/read:024ResidualRegress/bias2_reg/Adam_3/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_2:0$ResidualRegress/w3_reg/Adam_2/Assign$ResidualRegress/w3_reg/Adam_2/read:021ResidualRegress/w3_reg/Adam_2/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_3:0$ResidualRegress/w3_reg/Adam_3/Assign$ResidualRegress/w3_reg/Adam_3/read:021ResidualRegress/w3_reg/Adam_3/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_2:0'ResidualRegress/bias3_reg/Adam_2/Assign'ResidualRegress/bias3_reg/Adam_2/read:024ResidualRegress/bias3_reg/Adam_2/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_3:0'ResidualRegress/bias3_reg/Adam_3/Assign'ResidualRegress/bias3_reg/Adam_3/read:024ResidualRegress/bias3_reg/Adam_3/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_2:0$ResidualRegress/w4_reg/Adam_2/Assign$ResidualRegress/w4_reg/Adam_2/read:021ResidualRegress/w4_reg/Adam_2/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_3:0$ResidualRegress/w4_reg/Adam_3/Assign$ResidualRegress/w4_reg/Adam_3/read:021ResidualRegress/w4_reg/Adam_3/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_2:0'ResidualRegress/bias4_reg/Adam_2/Assign'ResidualRegress/bias4_reg/Adam_2/read:024ResidualRegress/bias4_reg/Adam_2/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_3:0'ResidualRegress/bias4_reg/Adam_3/Assign'ResidualRegress/bias4_reg/Adam_3/read:024ResidualRegress/bias4_reg/Adam_3/Initializer/zeros:0
\
beta1_power_6:0beta1_power_6/Assignbeta1_power_6/read:02beta1_power_6/initial_value:0
\
beta2_power_6:0beta2_power_6/Assignbeta2_power_6/read:02beta2_power_6/initial_value:0
 
ResidualRegress/w1_reg/Adam_4:0$ResidualRegress/w1_reg/Adam_4/Assign$ResidualRegress/w1_reg/Adam_4/read:021ResidualRegress/w1_reg/Adam_4/Initializer/zeros:0
 
ResidualRegress/w1_reg/Adam_5:0$ResidualRegress/w1_reg/Adam_5/Assign$ResidualRegress/w1_reg/Adam_5/read:021ResidualRegress/w1_reg/Adam_5/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_4:0'ResidualRegress/bias1_reg/Adam_4/Assign'ResidualRegress/bias1_reg/Adam_4/read:024ResidualRegress/bias1_reg/Adam_4/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_5:0'ResidualRegress/bias1_reg/Adam_5/Assign'ResidualRegress/bias1_reg/Adam_5/read:024ResidualRegress/bias1_reg/Adam_5/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_4:0$ResidualRegress/w2_reg/Adam_4/Assign$ResidualRegress/w2_reg/Adam_4/read:021ResidualRegress/w2_reg/Adam_4/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_5:0$ResidualRegress/w2_reg/Adam_5/Assign$ResidualRegress/w2_reg/Adam_5/read:021ResidualRegress/w2_reg/Adam_5/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_4:0'ResidualRegress/bias2_reg/Adam_4/Assign'ResidualRegress/bias2_reg/Adam_4/read:024ResidualRegress/bias2_reg/Adam_4/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_5:0'ResidualRegress/bias2_reg/Adam_5/Assign'ResidualRegress/bias2_reg/Adam_5/read:024ResidualRegress/bias2_reg/Adam_5/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_4:0$ResidualRegress/w3_reg/Adam_4/Assign$ResidualRegress/w3_reg/Adam_4/read:021ResidualRegress/w3_reg/Adam_4/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_5:0$ResidualRegress/w3_reg/Adam_5/Assign$ResidualRegress/w3_reg/Adam_5/read:021ResidualRegress/w3_reg/Adam_5/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_4:0'ResidualRegress/bias3_reg/Adam_4/Assign'ResidualRegress/bias3_reg/Adam_4/read:024ResidualRegress/bias3_reg/Adam_4/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_5:0'ResidualRegress/bias3_reg/Adam_5/Assign'ResidualRegress/bias3_reg/Adam_5/read:024ResidualRegress/bias3_reg/Adam_5/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_4:0$ResidualRegress/w4_reg/Adam_4/Assign$ResidualRegress/w4_reg/Adam_4/read:021ResidualRegress/w4_reg/Adam_4/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_5:0$ResidualRegress/w4_reg/Adam_5/Assign$ResidualRegress/w4_reg/Adam_5/read:021ResidualRegress/w4_reg/Adam_5/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_4:0'ResidualRegress/bias4_reg/Adam_4/Assign'ResidualRegress/bias4_reg/Adam_4/read:024ResidualRegress/bias4_reg/Adam_4/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_5:0'ResidualRegress/bias4_reg/Adam_5/Assign'ResidualRegress/bias4_reg/Adam_5/read:024ResidualRegress/bias4_reg/Adam_5/Initializer/zeros:0
\
beta1_power_7:0beta1_power_7/Assignbeta1_power_7/read:02beta1_power_7/initial_value:0
\
beta2_power_7:0beta2_power_7/Assignbeta2_power_7/read:02beta2_power_7/initial_value:0
 
ResidualRegress/w1_reg/Adam_6:0$ResidualRegress/w1_reg/Adam_6/Assign$ResidualRegress/w1_reg/Adam_6/read:021ResidualRegress/w1_reg/Adam_6/Initializer/zeros:0
 
ResidualRegress/w1_reg/Adam_7:0$ResidualRegress/w1_reg/Adam_7/Assign$ResidualRegress/w1_reg/Adam_7/read:021ResidualRegress/w1_reg/Adam_7/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_6:0'ResidualRegress/bias1_reg/Adam_6/Assign'ResidualRegress/bias1_reg/Adam_6/read:024ResidualRegress/bias1_reg/Adam_6/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_7:0'ResidualRegress/bias1_reg/Adam_7/Assign'ResidualRegress/bias1_reg/Adam_7/read:024ResidualRegress/bias1_reg/Adam_7/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_6:0$ResidualRegress/w2_reg/Adam_6/Assign$ResidualRegress/w2_reg/Adam_6/read:021ResidualRegress/w2_reg/Adam_6/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_7:0$ResidualRegress/w2_reg/Adam_7/Assign$ResidualRegress/w2_reg/Adam_7/read:021ResidualRegress/w2_reg/Adam_7/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_6:0'ResidualRegress/bias2_reg/Adam_6/Assign'ResidualRegress/bias2_reg/Adam_6/read:024ResidualRegress/bias2_reg/Adam_6/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_7:0'ResidualRegress/bias2_reg/Adam_7/Assign'ResidualRegress/bias2_reg/Adam_7/read:024ResidualRegress/bias2_reg/Adam_7/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_6:0$ResidualRegress/w3_reg/Adam_6/Assign$ResidualRegress/w3_reg/Adam_6/read:021ResidualRegress/w3_reg/Adam_6/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_7:0$ResidualRegress/w3_reg/Adam_7/Assign$ResidualRegress/w3_reg/Adam_7/read:021ResidualRegress/w3_reg/Adam_7/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_6:0'ResidualRegress/bias3_reg/Adam_6/Assign'ResidualRegress/bias3_reg/Adam_6/read:024ResidualRegress/bias3_reg/Adam_6/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_7:0'ResidualRegress/bias3_reg/Adam_7/Assign'ResidualRegress/bias3_reg/Adam_7/read:024ResidualRegress/bias3_reg/Adam_7/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_6:0$ResidualRegress/w4_reg/Adam_6/Assign$ResidualRegress/w4_reg/Adam_6/read:021ResidualRegress/w4_reg/Adam_6/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_7:0$ResidualRegress/w4_reg/Adam_7/Assign$ResidualRegress/w4_reg/Adam_7/read:021ResidualRegress/w4_reg/Adam_7/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_6:0'ResidualRegress/bias4_reg/Adam_6/Assign'ResidualRegress/bias4_reg/Adam_6/read:024ResidualRegress/bias4_reg/Adam_6/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_7:0'ResidualRegress/bias4_reg/Adam_7/Assign'ResidualRegress/bias4_reg/Adam_7/read:024ResidualRegress/bias4_reg/Adam_7/Initializer/zeros:0
\
beta1_power_8:0beta1_power_8/Assignbeta1_power_8/read:02beta1_power_8/initial_value:0
\
beta2_power_8:0beta2_power_8/Assignbeta2_power_8/read:02beta2_power_8/initial_value:0
 
ResidualRegress/w1_reg/Adam_8:0$ResidualRegress/w1_reg/Adam_8/Assign$ResidualRegress/w1_reg/Adam_8/read:021ResidualRegress/w1_reg/Adam_8/Initializer/zeros:0
 
ResidualRegress/w1_reg/Adam_9:0$ResidualRegress/w1_reg/Adam_9/Assign$ResidualRegress/w1_reg/Adam_9/read:021ResidualRegress/w1_reg/Adam_9/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_8:0'ResidualRegress/bias1_reg/Adam_8/Assign'ResidualRegress/bias1_reg/Adam_8/read:024ResidualRegress/bias1_reg/Adam_8/Initializer/zeros:0
Ź
"ResidualRegress/bias1_reg/Adam_9:0'ResidualRegress/bias1_reg/Adam_9/Assign'ResidualRegress/bias1_reg/Adam_9/read:024ResidualRegress/bias1_reg/Adam_9/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_8:0$ResidualRegress/w2_reg/Adam_8/Assign$ResidualRegress/w2_reg/Adam_8/read:021ResidualRegress/w2_reg/Adam_8/Initializer/zeros:0
 
ResidualRegress/w2_reg/Adam_9:0$ResidualRegress/w2_reg/Adam_9/Assign$ResidualRegress/w2_reg/Adam_9/read:021ResidualRegress/w2_reg/Adam_9/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_8:0'ResidualRegress/bias2_reg/Adam_8/Assign'ResidualRegress/bias2_reg/Adam_8/read:024ResidualRegress/bias2_reg/Adam_8/Initializer/zeros:0
Ź
"ResidualRegress/bias2_reg/Adam_9:0'ResidualRegress/bias2_reg/Adam_9/Assign'ResidualRegress/bias2_reg/Adam_9/read:024ResidualRegress/bias2_reg/Adam_9/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_8:0$ResidualRegress/w3_reg/Adam_8/Assign$ResidualRegress/w3_reg/Adam_8/read:021ResidualRegress/w3_reg/Adam_8/Initializer/zeros:0
 
ResidualRegress/w3_reg/Adam_9:0$ResidualRegress/w3_reg/Adam_9/Assign$ResidualRegress/w3_reg/Adam_9/read:021ResidualRegress/w3_reg/Adam_9/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_8:0'ResidualRegress/bias3_reg/Adam_8/Assign'ResidualRegress/bias3_reg/Adam_8/read:024ResidualRegress/bias3_reg/Adam_8/Initializer/zeros:0
Ź
"ResidualRegress/bias3_reg/Adam_9:0'ResidualRegress/bias3_reg/Adam_9/Assign'ResidualRegress/bias3_reg/Adam_9/read:024ResidualRegress/bias3_reg/Adam_9/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_8:0$ResidualRegress/w4_reg/Adam_8/Assign$ResidualRegress/w4_reg/Adam_8/read:021ResidualRegress/w4_reg/Adam_8/Initializer/zeros:0
 
ResidualRegress/w4_reg/Adam_9:0$ResidualRegress/w4_reg/Adam_9/Assign$ResidualRegress/w4_reg/Adam_9/read:021ResidualRegress/w4_reg/Adam_9/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_8:0'ResidualRegress/bias4_reg/Adam_8/Assign'ResidualRegress/bias4_reg/Adam_8/read:024ResidualRegress/bias4_reg/Adam_8/Initializer/zeros:0
Ź
"ResidualRegress/bias4_reg/Adam_9:0'ResidualRegress/bias4_reg/Adam_9/Assign'ResidualRegress/bias4_reg/Adam_9/read:024ResidualRegress/bias4_reg/Adam_9/Initializer/zeros:0
\
beta1_power_9:0beta1_power_9/Assignbeta1_power_9/read:02beta1_power_9/initial_value:0
\
beta2_power_9:0beta2_power_9/Assignbeta2_power_9/read:02beta2_power_9/initial_value:0
¤
 ResidualRegress/w1_reg/Adam_10:0%ResidualRegress/w1_reg/Adam_10/Assign%ResidualRegress/w1_reg/Adam_10/read:022ResidualRegress/w1_reg/Adam_10/Initializer/zeros:0
¤
 ResidualRegress/w1_reg/Adam_11:0%ResidualRegress/w1_reg/Adam_11/Assign%ResidualRegress/w1_reg/Adam_11/read:022ResidualRegress/w1_reg/Adam_11/Initializer/zeros:0
°
#ResidualRegress/bias1_reg/Adam_10:0(ResidualRegress/bias1_reg/Adam_10/Assign(ResidualRegress/bias1_reg/Adam_10/read:025ResidualRegress/bias1_reg/Adam_10/Initializer/zeros:0
°
#ResidualRegress/bias1_reg/Adam_11:0(ResidualRegress/bias1_reg/Adam_11/Assign(ResidualRegress/bias1_reg/Adam_11/read:025ResidualRegress/bias1_reg/Adam_11/Initializer/zeros:0
¤
 ResidualRegress/w2_reg/Adam_10:0%ResidualRegress/w2_reg/Adam_10/Assign%ResidualRegress/w2_reg/Adam_10/read:022ResidualRegress/w2_reg/Adam_10/Initializer/zeros:0
¤
 ResidualRegress/w2_reg/Adam_11:0%ResidualRegress/w2_reg/Adam_11/Assign%ResidualRegress/w2_reg/Adam_11/read:022ResidualRegress/w2_reg/Adam_11/Initializer/zeros:0
°
#ResidualRegress/bias2_reg/Adam_10:0(ResidualRegress/bias2_reg/Adam_10/Assign(ResidualRegress/bias2_reg/Adam_10/read:025ResidualRegress/bias2_reg/Adam_10/Initializer/zeros:0
°
#ResidualRegress/bias2_reg/Adam_11:0(ResidualRegress/bias2_reg/Adam_11/Assign(ResidualRegress/bias2_reg/Adam_11/read:025ResidualRegress/bias2_reg/Adam_11/Initializer/zeros:0
¤
 ResidualRegress/w3_reg/Adam_10:0%ResidualRegress/w3_reg/Adam_10/Assign%ResidualRegress/w3_reg/Adam_10/read:022ResidualRegress/w3_reg/Adam_10/Initializer/zeros:0
¤
 ResidualRegress/w3_reg/Adam_11:0%ResidualRegress/w3_reg/Adam_11/Assign%ResidualRegress/w3_reg/Adam_11/read:022ResidualRegress/w3_reg/Adam_11/Initializer/zeros:0
°
#ResidualRegress/bias3_reg/Adam_10:0(ResidualRegress/bias3_reg/Adam_10/Assign(ResidualRegress/bias3_reg/Adam_10/read:025ResidualRegress/bias3_reg/Adam_10/Initializer/zeros:0
°
#ResidualRegress/bias3_reg/Adam_11:0(ResidualRegress/bias3_reg/Adam_11/Assign(ResidualRegress/bias3_reg/Adam_11/read:025ResidualRegress/bias3_reg/Adam_11/Initializer/zeros:0
¤
 ResidualRegress/w4_reg/Adam_10:0%ResidualRegress/w4_reg/Adam_10/Assign%ResidualRegress/w4_reg/Adam_10/read:022ResidualRegress/w4_reg/Adam_10/Initializer/zeros:0
¤
 ResidualRegress/w4_reg/Adam_11:0%ResidualRegress/w4_reg/Adam_11/Assign%ResidualRegress/w4_reg/Adam_11/read:022ResidualRegress/w4_reg/Adam_11/Initializer/zeros:0
°
#ResidualRegress/bias4_reg/Adam_10:0(ResidualRegress/bias4_reg/Adam_10/Assign(ResidualRegress/bias4_reg/Adam_10/read:025ResidualRegress/bias4_reg/Adam_10/Initializer/zeros:0
°
#ResidualRegress/bias4_reg/Adam_11:0(ResidualRegress/bias4_reg/Adam_11/Assign(ResidualRegress/bias4_reg/Adam_11/read:025ResidualRegress/bias4_reg/Adam_11/Initializer/zeros:0
`
beta1_power_10:0beta1_power_10/Assignbeta1_power_10/read:02beta1_power_10/initial_value:0
`
beta2_power_10:0beta2_power_10/Assignbeta2_power_10/read:02beta2_power_10/initial_value:0

TrResidual/alpha/Adam:0TrResidual/alpha/Adam/AssignTrResidual/alpha/Adam/read:02)TrResidual/alpha/Adam/Initializer/zeros:0

TrResidual/alpha/Adam_1:0TrResidual/alpha/Adam_1/AssignTrResidual/alpha/Adam_1/read:02+TrResidual/alpha/Adam_1/Initializer/zeros:0
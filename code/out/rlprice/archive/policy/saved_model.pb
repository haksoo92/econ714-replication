??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
5ActorDistributionNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense/kernel
?
IActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	?d*
dtype0
?
3ActorDistributionNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*D
shared_name53ActorDistributionNetwork/EncodingNetwork/dense/bias
?
GActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3ActorDistributionNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0
?
@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias
?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpReadVariableOp@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias*
_output_shapes
:*
dtype0
?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*_
shared_namePNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel
?
bActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOpNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel*
_output_shapes

:d*
dtype0
?
LActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias
?
`ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpReadVariableOpLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
	3

4

0
 
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3ActorDistributionNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUENActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE

ref
1

_actor_network
z
_encoder
_projection_networks
	variables
trainable_variables
regularization_losses
	keras_api
n
_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
z
_means_projection_layer
	_bias
	variables
trainable_variables
regularization_losses
	keras_api
#
0
1
	2

3
4
#
0
1
	2

3
4
 
?
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses

$0
%1

0
1

0
1
 
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
h

	kernel

bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
\
bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api

	0

1
2

	0

1
2
 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
 

0
1
 
 
 
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

kernel
bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
 

$0
%1
 
 
 

	0

1

	0

1
 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
+	variables
,trainable_variables
-regularization_losses

0

0
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
/	variables
0trainable_variables
1regularization_losses
 

0
1
 
 
 
 
 
 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
8	variables
9trainable_variables
:regularization_losses

0
1

0
1
 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
<	variables
=trainable_variables
>regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0_discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????

action_0_observationPlaceholder*+
_output_shapes
:??????????*
dtype0* 
shape:??????????
j
action_0_rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0_step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_509197414
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_509197426
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_509197448
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_509197441
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpbActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOp`ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpConst*
Tin

2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_save_509197698
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference__traced_restore_509197726??
?
e
__inference_<lambda>_509197170!
readvariableop_resource:	 
identity	??ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
?
9
'__inference_get_initial_state_509197420

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
-__inference_function_with_signature_509197421

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_get_initial_state_509197420*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?t
?
+__inference_polymorphic_action_fn_509197524
	step_type

reward
discount
observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?d\
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dx
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:du
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:??????????
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:??????????
JActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
FActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDims?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0SActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddOActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
OActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
IActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0XActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:??????????
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:??????????
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:?
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
UNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_likeFilldNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Shape:output:0dNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Const:output:0*
T0*#
_output_shapes
:??????????
ONormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mulMul8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0^Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like:output:0*
T0*#
_output_shapes
:?????????W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic/sample/ShapeShapeSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastToSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:?????????u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:??????????
NoOpNoOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:??????????: : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:XT
+
_output_shapes
:??????????
%
_user_specified_nameobservation
?
9
'__inference_get_initial_state_509197652

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
"__inference__traced_save_509197698
file_prefix'
#savev2_variable_read_readvariableop	T
Psavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableop_
[savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopm
isavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopk
gsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableop[savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopisavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopgsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*@
_input_shapes/
-: : :	?d:d::d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	?d: 

_output_shapes
:d: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
?u
?
+__inference_polymorphic_action_fn_509197599
time_step_step_type
time_step_reward
time_step_discount
time_step_observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?d\
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dx
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:du
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:??????????
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:??????????
JActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
FActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDims?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0SActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddOActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
OActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
IActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0XActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:??????????
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:??????????
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:?
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
UNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_likeFilldNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Shape:output:0dNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Const:output:0*
T0*#
_output_shapes
:??????????
ONormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mulMul8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0^Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like:output:0*
T0*#
_output_shapes
:?????????W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic/sample/ShapeShapeSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastToSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:?????????u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:??????????
NoOpNoOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:??????????: : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:b^
+
_output_shapes
:??????????
/
_user_specified_nametime_step/observation
?t
?
+__inference_polymorphic_action_fn_509197381
	time_step
time_step_1
time_step_2
time_step_3`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?d\
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dx
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:du
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:??????????
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:??????????
JActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
FActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDims?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0SActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddOActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
OActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
IActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0XActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:??????????
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:??????????
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:?
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
UNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_likeFilldNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Shape:output:0dNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Const:output:0*
T0*#
_output_shapes
:??????????
ONormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mulMul8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0^Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like:output:0*
T0*#
_output_shapes
:?????????W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Deterministic/sample/ShapeShapeSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastToSNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:?????????u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:??????????
NoOpNoOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:??????????: : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:VR
+
_output_shapes
:??????????
#
_user_specified_name	time_step
?
?
'__inference_signature_wrapper_509197414
discount
observation

reward
	step_type
unknown:	?d
	unknown_0:d
	unknown_1:d
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *6
f1R/
-__inference_function_with_signature_509197394k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:??????????:?????????:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:??????????
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?
9
'__inference_signature_wrapper_509197426

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *6
f1R/
-__inference_function_with_signature_509197421*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
/
-__inference_function_with_signature_509197444?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_<lambda>_509197173*(
_construction_contextkEagerRuntime*
_input_shapes 
_
 
__inference_<lambda>_509197173*(
_construction_contextkEagerRuntime*
_input_shapes 
?
g
'__inference_signature_wrapper_509197441
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *6
f1R/
-__inference_function_with_signature_509197433^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
-__inference_function_with_signature_509197394
	step_type

reward
discount
observation
unknown:	?d
	unknown_0:d
	unknown_1:d
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_polymorphic_action_fn_509197381k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:??????????
'
_user_specified_name0/observation
?X
?
1__inference_polymorphic_distribution_fn_509197649
	step_type

reward
discount
observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?d\
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:dx
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:du
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:?????????{
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:??????????
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:??????????
JActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
FActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDims?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0SActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddOActorDistributionNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
OActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
QActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
IActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0XActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:??????????
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:??????????
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:?
[Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
UNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_likeFilldNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Shape:output:0dNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like/Const:output:0*
T0*#
_output_shapes
:??????????
ONormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mulMul8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0^Normal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/ones_like:output:0*
T0*#
_output_shapes
:?????????W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_1IdentitySNormal_CONSTRUCTED_AT_ActorDistributionNetwork_NormalProjectionNetwork/mode/mul:z:0^NoOp*
T0*#
_output_shapes
:?????????[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:??????????: : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:XT
+
_output_shapes
:??????????
%
_user_specified_nameobservation
?
m
-__inference_function_with_signature_509197433
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference_<lambda>_509197170^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
? 
?
%__inference__traced_restore_509197726
file_prefix#
assignvariableop_variable:	 [
Hassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernel:	?dT
Fassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_bias:da
Sassignvariableop_3_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias:s
aassignvariableop_4_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel:dm
_assignvariableop_5_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias:

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpHassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpFassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpSassignvariableop_3_actordistributionnetwork_normalprojectionnetwork_bias_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpaassignvariableop_4_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp_assignvariableop_5_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
)
'__inference_signature_wrapper_509197448?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *6
f1R/
-__inference_function_with_signature_509197444*(
_construction_contextkEagerRuntime*
_input_shapes "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0_discount:0?????????
B
0/observation1
action_0_observation:0??????????
0
0/reward$
action_0_reward:0?????????
6
0/step_type'
action_0_step_type:0?????????6
action,
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:?i
?

train_step
metadata
model_variables
_all_assets

signatures

Taction
Udistribution
Vget_initial_state
Wget_metadata
Xget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
D
0
1
2
	3

4"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

Yaction
Zget_initial_state
[get_train_step
\get_metadata"
signature_map
H:F	?d25ActorDistributionNetwork/EncodingNetwork/dense/kernel
A:?d23ActorDistributionNetwork/EncodingNetwork/dense/bias
N:L2@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias
`:^d2NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel
Z:X2LActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias
1
ref
1"
trackable_tuple_wrapper
2
_actor_network"
_generic_user_object
?
_encoder
_projection_networks
	variables
trainable_variables
regularization_losses
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_means_projection_layer
	_bias
	variables
trainable_variables
regularization_losses
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
C
0
1
	2

3
4"
trackable_list_wrapper
C
0
1
	2

3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?

	kernel

bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
?
bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
5
	0

1
2"
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
+	variables
,trainable_variables
-regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
/	variables
0trainable_variables
1regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
8	variables
9trainable_variables
:regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
<	variables
=trainable_variables
>regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
+__inference_polymorphic_action_fn_509197524
+__inference_polymorphic_action_fn_509197599?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_polymorphic_distribution_fn_509197649?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_get_initial_state_509197652?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_509197173"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_509197170"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_509197414
0/discount0/observation0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_509197426
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_509197441"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_509197448"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecU
argsM?J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecU
argsM?J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec?
args7?4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec?
args7?4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 =
__inference_<lambda>_509197170?

? 
? "? 	6
__inference_<lambda>_509197173?

? 
? "? T
'__inference_get_initial_state_509197652)"?
?
?

batch_size 
? "? ?
+__inference_polymorphic_action_fn_509197524?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????8
observation)?&
observation??????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
+__inference_polymorphic_action_fn_509197599?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount?????????B
observation3?0
time_step/observation??????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
1__inference_polymorphic_distribution_fn_509197649?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????8
observation)?&
observation??????????
? 
? "???

PolicyStep?
action??????
`
B??

atol? 

loc??????????

rtol? 
J?G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
?
j
parameters
? 
?
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
state? 
info? ?
'__inference_signature_wrapper_509197414?	
???
? 
???
.

0/discount ?

0/discount?????????
<
0/observation+?(
0/observation??????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"+?(
&
action?
action?????????b
'__inference_signature_wrapper_50919742670?-
? 
&?#
!

batch_size?

batch_size "? [
'__inference_signature_wrapper_5091974410?

? 
? "?

int64?
int64 	?
'__inference_signature_wrapper_509197448?

? 
? "? 
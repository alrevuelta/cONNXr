#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onnx.pb-c.h"
#include "embeddedml_debug.h"
#include "embeddedml_operators.h"

/* Important Notes
 *  If you are contributing by implementing an operator, make sure that you follow
 *  onnx specifications in https://github.com/onnx/onnx/blob/master/docs/Operators.md
 *  Remember to copy the documentation, like inputs, outputs and type constraints.
 *  See previosly implemented operators as example.
 */

// Template example
 /*! \fn COPY_PASTE_FUNCTION_DECLARATION
  *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
  *
  *         Limitations: There might be some limitations with respect to the onnx
  *           official operator. Write here possible limitations, i.e. if the
  *           function doesnt work with all types, or if it works with a specific
  *           number of dimensions only
  *  \param[in]  xx xx
  *  \param[in]  xx xx
  *  \param[out] xx xx
  *  \return     xx
  */
void Operators_Abs(void *todo)
{

}

void Operators_Acos(void *todo)
{

}

void Operators_Acosh(void *todo)
{

}

void Operators_Add(void *inOut, void *matrix, int m, enum _Onnx__TensorProto__DataType type)
{
  // TODO Using float by default
/*
  while (m > 0) {
    m--;
    inOut[m] = inOut[m] + matrix[m];
  }*/
}

void Operators_And(void *todo)
{

}
void Operators_ArgMax(void *x, int dimx, int dimy, int axis, int keepdims, int* out)
{
  // TODO keepdims is not used
  // TODO Only axis=1 is supported
  // Only 2D are supported
  float *xf = (float*)x;
  for (int i = 0; i < dimx; i++) {
    int argmaxindex = 0;
    float maxvalue = xf[i*dimy];
    for (int j = 0; j < dimy; j++) {
      if (xf[j+i*dimy] > maxvalue) {
        maxvalue = xf[j+i*dimy];
        argmaxindex = j;
      }
    }
    out[i] = argmaxindex;
  }
}
void Operators_ArgMin(void *todo)
{

}
void Operators_Asin(void *todo)
{

}
void Operators_Asinh(void *todo)
{

}
void Operators_Atan(void *todo)
{

}
void Operators_Atanh(void *todo)
{

}
void Operators_AveragePool(void *todo)
{

}
void Operators_BatchNormalization(void *todo)
{

}
void Operators_BitShift(void *todo)
{

}
void Operators_Cast(void *in, void *out, enum _Onnx__TensorProto__DataType inType, enum _Onnx__TensorProto__DataType outType)
{

}
void Operators_Ceil(void *todo)
{

}
void Operators_Clip(void *todo)
{

}
void Operators_Compress(void *todo)
{

}
void Operators_Concat(void *todo)
{

}
void Operators_ConcatFromSequence(void *todo)
{

}
void Operators_Constant(void *todo)
{

}
void Operators_ConstantOfShape(void *todo)
{

}
void Operators_Conv(void *todo)
{

}
void Operators_ConvInteger(void *todo)
{

}
void Operators_ConvTranspose(void *todo)
{

}
void Operators_Cos(void *todo)
{

}
void Operators_Cosh(void *todo)
{

}
void Operators_CumSum(void *todo)
{

}
void Operators_DepthToSpace(void *todo)
{

}
void Operators_DequantizeLinear(void *todo)
{

}
void Operators_Det(void *todo)
{

}
void Operators_Div(void *todo)
{

}
void Operators_Dropout(void *todo)
{

}
void Operators_Elu(void *todo)
{

}
void Operators_Equal(void *todo)
{

}
void Operators_Erf(void *todo)
{

}
void Operators_Exp(void *todo)
{

}
void Operators_Expand(void *todo)
{

}
void Operators_EyeLike(void *todo)
{

}
void Operators_Flatten(void *todo)
{

}
void Operators_Floor(void *todo)
{

}
void Operators_GRU(void *todo)
{

}
void Operators_Gather(void *todo)
{

}
void Operators_GatherElements(void *todo)
{

}
void Operators_GatherND(void *todo)
{

}
void Operators_Gemm(void *todo)
{

}
void Operators_GlobalAveragePool(void *todo)
{

}
void Operators_GlobalLpPool(void *todo)
{

}
void Operators_GlobalMaxPool(void *todo)
{

}
void Operators_Greater(void *todo)
{

}
void Operators_HardSigmoid(void *todo)
{

}
void Operators_Hardmax(void *todo)
{

}
void Operators_Identity(void *todo)
{

}
void Operators_If(void *todo)
{

}
void Operators_InstanceNormalization(void *todo)
{

}
void Operators_IsInf(void *todo)
{

}
void Operators_IsNaN(void *todo)
{

}
void Operators_LRN(void *todo)
{

}
void Operators_LSTM(void *todo)
{

}
void Operators_LeakyRelu(void *todo)
{

}
void Operators_Less(void *todo)
{

}
void Operators_Log(void *todo)
{

}
void Operators_LogSoftmax(void *todo)
{

}
void Operators_Loop(void *todo)
{

}
void Operators_LpNormalization(void *todo)
{

}
void Operators_LpPool(void *todo)
{

}

/*! \fn void Operators_MatMul(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c)
 *  \brief MatMul: Matrix product that behaves like numpy.matmul:
 *                 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
 *         Version: This version of the operator has been available since
 *                  version 9 of the default ONNX operator set. Other versions
 *                  of this operator: MatMul-1
 *         Inputs
 *          A : T. N-dimensional matrix A
 *          B : T. N-dimensional matrix B
 *         Outputs
 *          Y : T. Matrix multiply results from A * B
 *         Type Constraints
 *          T : tensor(float16), tensor(float), tensor(double), tensor(uint32),
 *              tensor(uint64), tensor(int32), tensor(int64)
 *              Constrain input and output types to float/int tensors.
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  Onnx__TensorProto a
 *  \param[in]  Onnx__TensorProto b
 *  \param[out] Onnx__TensorProto c
 *  \return     void
 */
void Operators_MatMul(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *o)
{
  DEBUG_PRINT("Calling Operators_MatMul");
  // Naive 2x2 matrix mult
  // Allocte memory
  o->dims = malloc(2 * sizeof(int64_t));
  o->float_data = malloc(a->dims[0] * b->dims[1] * sizeof(float));
  o->name = malloc(30 * sizeof(char));

  // Populate some parameters
  o->n_dims = 2;
  o->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT; //hardcoded
  o->n_float_data = a->dims[0] * b->dims[1];
  o->name = "todo_set_name\0";
  o->dims[0] = a->dims[0];
  o->dims[1] = b->dims[1];
  o->has_raw_data = 0;

  for (int i = 0; i < a->dims[0]; i++) {
    for (int j = 0; j < b->dims[1]; j++) {
      float sum = 0;
      for (int p = 0; p < a->dims[1]; p++) {
        sum += (a->float_data[i*a->dims[1]+p] * b->float_data[p*b->dims[1]+j]);
        // Saturate the value?
      }
      o->float_data[i*b->dims[1]+j] = sum;
    }
  }

  Debug_PrintTensorProto(o);

  // Remove this, for testing
  for (int i = 0; i < o->n_float_data; i++)
  {
    printf("%f\n", o->float_data[i]);
  }


  /*
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0;
      for (int p = 0; p < n; p++) {
        sum += (a[i*n+p] * b[p*k+j]);
        // Saturate the value?
      }
      c[i*k+j] = sum;
    }
  }*/
  /*
  switch(type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      break;
    default:
      break;
  }*/
}

void Operators_MatMulInteger(void *todo)
{

}
void Operators_Max(void *todo)
{

}
void Operators_MaxPool(void *todo)
{

}
void Operators_MaxRoiPool(void *todo)
{

}
void Operators_MaxUnpool(void *todo)
{

}
void Operators_Mean(void *todo)
{

}
void Operators_Min(void *todo)
{

}
void Operators_Mod(void *todo)
{

}
void Operators_Mul(void *todo)
{

}
void Operators_Multinomial(void *todo)
{

}
void Operators_Neg(void *todo)
{

}
void Operators_NonMaxSuppression(void *todo)
{

}
void Operators_NonZero(void *todo)
{

}
void Operators_Not(void *todo)
{

}
void Operators_OneHot(void *todo)
{

}
void Operators_Or(void *todo)
{

}
void Operators_PRelu(void *todo)
{

}
void Operators_Pad(void *todo)
{

}
void Operators_Pow(void *todo)
{

}
void Operators_QLinearConv(void *todo)
{

}
void Operators_QLinearMatMul(void *todo)
{

}
void Operators_QuantizeLinear(void *todo)
{

}
void Operators_RNN(void *todo)
{

}
void Operators_RandomNormal(void *todo)
{

}
void Operators_RandomNormalLike(void *todo)
{

}
void Operators_RandomUniform(void *todo)
{

}
void Operators_RandomUniformLike(void *todo)
{

}
void Operators_Reciprocal(void *todo)
{

}
void Operators_ReduceL1(void *todo)
{

}
void Operators_ReduceL2(void *todo)
{

}
void Operators_ReduceLogSum(void *todo)
{

}
void Operators_ReduceLogSumExp(void *todo)
{

}
void Operators_ReduceMax(void *todo)
{

}
void Operators_ReduceMean(void *todo)
{

}
void Operators_ReduceMin(void *todo)
{

}
void Operators_ReduceProd(void *todo)
{

}
void Operators_ReduceSum(void *todo)
{

}
void Operators_ReduceSumSquare(void *todo)
{

}
void Operators_Relu(float *inOut, int size)
{
  for (int i = 0; i < size; i++)
  {
    if (inOut[i] < 0)
    {
      inOut[i] = 0;
    }
  }
}

// tx/ty are the initial tensor dimensions
// ox/oy are the output tensor dimensions
// A reshape doesnt modify the input tensor. The reshape is performed
// by reinterpreting the dimensions
void Operators_Reshape(float *t, int tx, int ty, int ox, int oy)
{
  // Dummy func
}
void Operators_Resize(void *todo)
{

}
void Operators_ReverseSequence(void *todo)
{

}
void Operators_RoiAlign(void *todo)
{

}
void Operators_Round(void *todo)
{

}
void Operators_Scan(void *todo)
{

}
void Operators_Scatter(void *todo)
{

}
void Operators_ScatterElements(void *todo)
{

}
void Operators_ScatterND(void *todo)
{

}
void Operators_Selu(void *todo)
{

}
void Operators_SequenceAt(void *todo)
{

}
void Operators_SequenceConstruct(void *todo)
{

}
void Operators_SequenceEmpty(void *todo)
{

}
void Operators_SequenceErase(void *todo)
{

}
void Operators_SequenceInsert(void *todo)
{

}
void Operators_SequenceLength(void *todo)
{

}
void Operators_Shape(void *todo)
{

}
void Operators_Shrink(void *todo)
{

}
void Operators_Sigmoid(void *x, int size)
{
  float *xf = (float*)x;
  while (size > 0) {
    size--;
    xf[size] = (1/(1 + exp(-(xf[size]))));
  }
}
void Operators_Sign(void *todo)
{

}
void Operators_Sin(void *todo)
{

}
void Operators_Sinh(void *todo)
{

}
void Operators_Size(void *todo)
{

}
void Operators_Slice(void *todo)
{

}

// Works with 1 dimension.
void Operators_Softmax(void *x, int dimx, int dimy)
{
  // TODO Use dimy to work with 2 dimensions.
  float sumExp = 0;
  float *xf = (float*) x;
  for (int i = 0; i < dimx; i++) {
    sumExp += exp(xf[i]);
  }

  for (int i = 0; i < dimx; i++) {
    xf[i] = exp(xf[i])/sumExp;
  }
}
void Operators_Softplus(void *todo)
{

}
void Operators_Softsign(void *todo)
{

}
void Operators_SpaceToDepth(void *todo)
{

}
void Operators_Split(void *todo)
{

}
void Operators_SplitToSequence(void *todo)
{

}
void Operators_Sqrt(void *todo)
{

}
void Operators_Squeeze(void *todo)
{

}
void Operators_StringNormalizer(void *todo)
{

}
void Operators_Sub(void *todo)
{

}
void Operators_Sum(void *todo)
{

}
void Operators_Tan(void *todo)
{

}
void Operators_Tanh(void *todo)
{

}
void Operators_TfIdfVectorizer(void *todo)
{

}
void Operators_ThresholdedRelu(void *todo)
{

}
void Operators_Tile(void *todo)
{

}
void Operators_TopK(void *todo)
{

}
void Operators_Transpose(void *todo)
{

}
void Operators_Unique(void *todo)
{

}
void Operators_Unsqueeze(void *todo)
{

}
void Operators_Upsample(void *todo)
{

}
void Operators_Where(void *todo)
{

}
void Operators_Xor(void *todo)
{

}

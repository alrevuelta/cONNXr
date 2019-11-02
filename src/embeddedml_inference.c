#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_utils.h"
#include "embeddedml_debug.h"
#include "embeddedml_outputs.h"
#include "embeddedml_operators.h"

// Investigate what to do with the output. Is it always a set of TensorProto?
int inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  int error = 0;
  DEBUG_PRINT("Calling inferenceFloat");
  DEBUG_PRINT("The graph has nodes=%zu", model->graph->n_node);

  // TODO Check that number of inputs provided match the model ones

  // Iterate all nodes in the graph
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    char *operation = model->graph->node[nodeIdx]->op_type;
    DEBUG_PRINT("node=%d, operation=%s, n_input=%zu, n_output=%zu",
                nodeIdx,
                model->graph->node[nodeIdx]->op_type,
                model->graph->node[nodeIdx]->n_input,
                model->graph->node[nodeIdx]->n_output);

/*
    for (int i = 0; i < model->graph->n_node; i++)
    {
      printf("model->graph->node[%d]->n_input %zu\n", i, model->graph->node[i]->n_input);
      for (int j = 0; j < model->graph->node[i]->n_input; j++) {
        printf("model->graph->node[%d]->input[%d] %s\n", i, j, model->graph->node[i]->input[j]);
      }
      printf("model->graph->node[%d]->n_output %zu\n", i, model->graph->node[i]->n_output);
      for (int j = 0; j < model->graph->node[i]->n_output; j++) {
        printf("model->graph->node[%d]->output[%d] %s\n", i, j, model->graph->node[i]->output[j]);
      }
      printf("model->graph->node[%d]->name %s\n", i, model->graph->node[i]->name);
      printf("model->graph->node[%d]->op_type %s\n", i, model->graph->node[i]->op_type);
    }*/


  /* This is a bit ugly, just a first draft. The idea in the future is that
   * on compile time the model is known, so only the needed operators are
   * compiled as part of the executable. Since this is targeted to small
   * devices, size is very relevant */
    if (!strcmp(operation, "Abs"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Acos"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Acosh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Add"))
    {
      // Inputs are deterministic given the operation type. Add:
      // * A: T. First operand
      // * B: T. Second operand
      DEBUG_PRINT("operation=%s, input=0 first operand name %s", operation, model->graph->node[nodeIdx]->input[0]);
      DEBUG_PRINT("operation=%s, input=1 second operand name %s", operation, model->graph->node[nodeIdx]->input[1]);

      Onnx__TensorProto *result;



      // Maybe the output tensor should be allocated inside the operator function.
      /*outputs_allocateOneTensor(result,...);
                                int32_t data_type,
                                int *dimensions,
                                int nDims);*/


/*
      Onnx__TensorProto *operandA = searchTensorInInitializers(
                                          model,
                                          model->graph->node[nodeIdx]->input[0]);
      Onnx__TensorProto *operandB = searchTensorInInitializers(
                                          model,
                                          model->graph->node[nodeIdx]->input[1]);*/

/* TODO
      Operators_Add(input,
                    tensor->float_data,
                    tensor->dims[0],
                    tensor->data_type);*/
    }
    else if (!strcmp(operation, "And"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ArgMax"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ArgMin"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Asin"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Asinh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Atan"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Atanh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "AveragePool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "BatchNormalization"))
    {
       // TODO
    }
    else if (!strcmp(operation, "BitShift"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Cast"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Ceil"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Clip"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Compress"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Concat"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ConcatFromSequence"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Constant"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ConstantOfShape"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Conv"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ConvInteger"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ConvTranspose"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Cos"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Cosh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "CumSum"))
    {
       // TODO
    }
    else if (!strcmp(operation, "DepthToSpace"))
    {
       // TODO
    }
    else if (!strcmp(operation, "DequantizeLinear"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Det"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Div"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Dropout"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Elu"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Equal"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Erf"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Exp"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Expand"))
    {
       // TODO
    }
    else if (!strcmp(operation, "EyeLike"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Flatten"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Floor"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GRU"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Gather"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GatherElements"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GatherND"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Gemm"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GlobalAveragePool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GlobalLpPool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "GlobalMaxPool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Greater"))
    {
       // TODO
    }
    else if (!strcmp(operation, "HardSigmoid"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Hardmax"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Identity"))
    {
       // TODO
    }
    else if (!strcmp(operation, "If"))
    {
       // TODO
    }
    else if (!strcmp(operation, "InstanceNormalization"))
    {
       // TODO
    }
    else if (!strcmp(operation, "IsInf"))
    {
       // TODO
    }
    else if (!strcmp(operation, "IsNaN"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LRN"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LSTM"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LeakyRelu"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Less"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Log"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LogSoftmax"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Loop"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LpNormalization"))
    {
       // TODO
    }
    else if (!strcmp(operation, "LpPool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "MatMul"))
    {
      // Inputs are deterministic given the operation type. MatMul:
      // * A: T. N-dimensional matrix A
      // * B: T. N-dimensional matrix B
      DEBUG_PRINT("operation=%s, input=0 first matrix %s", operation, model->graph->node[nodeIdx]->input[0]);
      DEBUG_PRINT("operation=%s, input=1 second matrix %s", operation, model->graph->node[nodeIdx]->input[1]);

      // TODO Maybe add some checks that input[0/1] is not null?

      Onnx__TensorProto *a = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[0]);
      Onnx__TensorProto *b = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[1]);

      DEBUG_PRINT("Running MatMul operator");
      Onnx__TensorProto *o = malloc (sizeof(*o));
      Operators_MatMul(a, b, o);
      DEBUG_PRINT("Printing ndims o = %zu", o->n_dims);
      DEBUG_PRINT("Printing name %s", o->name);
    }
    else if (!strcmp(operation, "MatMulInteger"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Max"))
    {
       // TODO
    }
    else if (!strcmp(operation, "MaxPool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "MaxRoiPool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "MaxUnpool"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Mean"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Min"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Mod"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Mul"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Multinomial"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Neg"))
    {
       // TODO
    }
    else if (!strcmp(operation, "NonMaxSuppression"))
    {
       // TODO
    }
    else if (!strcmp(operation, "NonZero"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Not"))
    {
       // TODO
    }
    else if (!strcmp(operation, "OneHot"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Or"))
    {
       // TODO
    }
    else if (!strcmp(operation, "PRelu"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Pad"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Pow"))
    {
       // TODO
    }
    else if (!strcmp(operation, "QLinearConv"))
    {
       // TODO
    }
    else if (!strcmp(operation, "QLinearMatMul"))
    {
       // TODO
    }
    else if (!strcmp(operation, "QuantizeLinear"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RNN"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RandomNormal"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RandomNormalLike"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RandomUniform"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RandomUniformLike"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Reciprocal"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceL1"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceL2"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceLogSum"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceLogSumExp"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceMax"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceMean"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceMin"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceProd"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceSum"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReduceSumSquare"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Relu"))
    {
      //Operators_Relu(input, inputDim);
    }
    else if (!strcmp(operation, "Reshape"))
    {
      // Inputs are deterministic given the operation type. Reshape:
      // * data: T. An input tensor
      // * shape: tensor (int64) Specified shape for output
      DEBUG_PRINT("operation=%s, input=0 tensor name %s", operation, model->graph->node[nodeIdx]->input[0]);
      DEBUG_PRINT("operation=%s, input=1 shape for output %s", operation, model->graph->node[nodeIdx]->input[1]);

/*
      Onnx__TensorProto *inputTensor = searchTensorInInitializers(
                                          model,
                                          model->graph->node[nodeIdx]->input[0]);
      Onnx__TensorProto *dimensions = searchTensorInInitializers(
                                          model,
                                          model->graph->node[nodeIdx]->input[1]);*/
      //Operators_Reshape(float *t, int tx, int ty, int ox, int oy);
    }
    else if (!strcmp(operation, "Resize"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ReverseSequence"))
    {
       // TODO
    }
    else if (!strcmp(operation, "RoiAlign"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Round"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Scan"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Scatter"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ScatterElements"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ScatterND"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Selu"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceAt"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceConstruct"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceEmpty"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceErase"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceInsert"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SequenceLength"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Shape"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Shrink"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sigmoid"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sign"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sin"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sinh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Size"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Slice"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Softmax"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Softplus"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Softsign"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SpaceToDepth"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Split"))
    {
       // TODO
    }
    else if (!strcmp(operation, "SplitToSequence"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sqrt"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Squeeze"))
    {
       // TODO
    }
    else if (!strcmp(operation, "StringNormalizer"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sub"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Sum"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Tan"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Tanh"))
    {
       // TODO
    }
    else if (!strcmp(operation, "TfIdfVectorizer"))
    {
       // TODO
    }
    else if (!strcmp(operation, "ThresholdedRelu"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Tile"))
    {
       // TODO
    }
    else if (!strcmp(operation, "TopK"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Transpose"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Unique"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Unsqueeze"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Upsample"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Where"))
    {
       // TODO
    }
    else if (!strcmp(operation, "Xor"))
    {
       // TODO
    }
    else{
      error = 1;
      return error;
    }
  }

  // TODO:
  // Free calculaterTensors memory
  // Free also extra allocations within the structure (i.e. doubles...)

  return error;
}

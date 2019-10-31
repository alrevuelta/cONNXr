#ifndef EMBEDDEDML_OUTPUTS_H
#define EMBEDDEDML_OUTPUTS_H

#include "onnx.pb-c.h"

#define MAX_TENSORS_BUFFER_SIZE

// This stores all the calculated tensors that are the output of each node. Many
// fields might be unused but the TensorProto type can be reused

// To simplify, a fixed buffer is used, but this must be allocated dynamically
// in the future

static Onnx__TensorProto calculaterTensors[MAX_TENSORS_BUFFER_SIZE];
static int tensorIdx;

void outputs_allocAllTensors();
void outputs_freeAllTensors();

void outputs_allocateOneTensor(Onnx__TensorProto *tpToAllocate,
                               int32_t data_type,
                               int *dimensions, int nDims);

Onnx__TensorProto *outputs_searchByName(char *name);
int outputs_addNewOutput(Onnx__TensorProto *tpToAdd);

#endif

#ifndef OPERATORS_H
#define OPERATORS_H
#include "../pb/onnx.pb-c.h"

int operator_add(operator__context *context);

int operator_argmax(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_batchnormalization(size_t n_input,
                                Onnx__TensorProto **input,
                                size_t n_attribute,
                                Onnx__AttributeProto **attribute,
                                size_t n_output,
                                Onnx__TensorProto **output);

int operator_cast(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_conv(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_leakyrelu(size_t n_input,
                       Onnx__TensorProto **input,
                       size_t n_attribute,
                       Onnx__AttributeProto **attribute,
                       size_t n_output,
                       Onnx__TensorProto **output);

int operator_matmul(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_maxpool(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_mul(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output);

int operator_relu(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_reshape(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_sigmoid(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_softmax(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_zipmap(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_sigmoid(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_softmax(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_zipmap(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_quantizelinear(size_t n_input,
                            Onnx__TensorProto **input,
                            size_t n_attribute,
                            Onnx__AttributeProto **attribute,
                            size_t n_output,
                            Onnx__TensorProto **output);

int operator_convinteger(size_t n_input,
                         Onnx__TensorProto **input,
                         size_t n_attribute,
                         Onnx__AttributeProto **attribute,
                         size_t n_output,
                         Onnx__TensorProto **output);

int operator_matmulinteger(size_t n_input,
                           Onnx__TensorProto **input,
                           size_t n_attribute,
                           Onnx__AttributeProto **attribute,
                           size_t n_output,
                           Onnx__TensorProto **output);


/* Quick prototyping
 * Hardcoded stuff that would be generated
 * using Python. Everything is thought to run
 * the MNIST model as a proof of concept, hence
 * only the operators that belong to that model are implemented */
typedef int (*onnx_operator)(
  operator__context *context
);

/* Add Operator Structs*/
typedef struct operator__onnx__add__input{
  Onnx__TensorProto *A;
  Onnx__TensorProto *B;
} operator__onnx__add__input;

typedef struct operator__onnx__add__output{
  Onnx__TensorProto *C;
} operator__onnx__add__output;

typedef struct operator__onnx__add__attribute{
  //No attributes
} operator__onnx__add__attribute;

typedef struct operator__onnx__add__context{
  struct operator__onnx__add__input *in;
  struct operator__onnx__add__output *out;
  struct operator__onnx__add__attribute *attr;
  onnx_operator run;
} operator__onnx__add__context;

/* Conv operator */
typedef struct operator__onnx__conv__input{
  Onnx__TensorProto *X;
  Onnx__TensorProto *W;
  Onnx__TensorProto *B;
} operator__onnx__conv__input;

typedef struct operator__onnx__conv__output{
  Onnx__TensorProto *Y;
} operator__onnx__conv__output;

typedef struct operator__onnx__conv__attribute{
  Onnx__AttributeProto *auto_pad;
  Onnx__AttributeProto *dilations;
  Onnx__AttributeProto *group;
  Onnx__AttributeProto *kernel_shape;
  Onnx__AttributeProto *pads;
  Onnx__AttributeProto *strides;
} operator__onnx__conv__attribute;

typedef struct operator__onnx__conv__context{
  struct operator__onnx__conv__input *in;
  struct operator__onnx__conv__output *out;
  struct operator__onnx__conv__attribute *attr;
  onnx_operator run;
} operator__onnx__conv__context;



/* Relu operator */
typedef struct operator__onnx__relu__input{
  Onnx__TensorProto *X;
} operator__onnx__relu__input;

typedef struct operator__onnx__relu__output{
  Onnx__TensorProto *Y;
} operator__onnx__relu__output;

typedef struct operator__onnx__relu__attribute{
  // Not attributes
} operator__onnx__relu__attribute;

typedef struct operator__onnx__relu__context{
  struct operator__onnx__relu__input *in;
  struct operator__onnx__relu__output *out;
  struct operator__onnx__relu__attribute *attr;
  onnx_operator run;
} operator__onnx__relu__context;



/* Maxpool operator */
typedef struct operator__onnx__maxpool__input{
  Onnx__TensorProto *X;
} operator__onnx__maxpool__input;

typedef struct operator__onnx__maxpool__output{
  Onnx__TensorProto *Y;
  Onnx__TensorProto *Indices;
} operator__onnx__maxpool__output;

typedef struct operator__onnx__maxpool__attribute{
  Onnx__AttributeProto *auto_pad;
  Onnx__AttributeProto *ceil_mode;
  Onnx__AttributeProto *dilations;
  Onnx__AttributeProto *kernel_shape;
  Onnx__AttributeProto *pads;
  Onnx__AttributeProto *storage_order;
  Onnx__AttributeProto *strides;
} operator__onnx__maxpool__attribute;

typedef struct operator__onnx__maxpool__context{
  struct operator__onnx__maxpool__input *in;
  struct operator__onnx__maxpool__output *out;
  struct operator__onnx__maxpool__attribute *attr;
  onnx_operator run;
} operator__onnx__maxpool__context;


/* Reshape operator */
typedef struct operator__onnx__reshape__input{
  Onnx__TensorProto *data;
  Onnx__TensorProto *shape;
} operator__onnx__reshape__input;

typedef struct operator__onnx__reshape__output{
  Onnx__TensorProto *reshaped;
} operator__onnx__reshape__output;

typedef struct operator__onnx__reshape__attribute{
  // Not attributes
} operator__onnx__reshape__attribute;

typedef struct operator__onnx__reshape__context{
  struct operator__onnx__reshape__input *in;
  struct operator__onnx__reshape__output *out;
  struct operator__onnx__reshape__attribute *attr;
  onnx_operator run;
} operator__onnx__reshape__context;



/* Matmul */
typedef struct operator__onnx__matmul__input{
  Onnx__TensorProto *A;
  Onnx__TensorProto *B;
} operator__onnx__matmul__input;

typedef struct operator__onnx__matmul__output{
  Onnx__TensorProto *Y;
} operator__onnx__matmul__output;

typedef struct operator__onnx__matmul__attribute{
  // No attributes
} operator__onnx__matmul__attribute;

typedef struct operator__onnx__matmul__context{
  struct operator__onnx__matmul__input *in;
  struct operator__onnx__matmul__output *out;
  struct operator__onnx__matmul__attribute *attr;
  onnx_operator run;
} operator__onnx__matmul__context;

typedef struct operator__context {
    Onnx__TensorProto **inputs;
    Onnx__AttributeProto **attributes;
    Onnx__TensorProto **outputs;
    onnx_operator run;
} operator__context;


/* Template */
/*
typedef struct operator__onnx__XXX__input{
  Onnx__TensorProto *XXX;
} operator__onnx__XXX__input;

typedef struct operator__onnx__XXX__output{
  Onnx__TensorProto *XXX;
} operator__onnx__XXX__output;

typedef struct operator__onnx__XXX__attribute{
  Onnx__AttributeProto *XXX;
} operator__onnx__XXX__attribute;

typedef struct operator__onnx__XXX__context{
  struct operator__onnx__XXX__input *in;
  struct operator__onnx__XXX__output *out;
  struct operator__onnx__XXX__attribute *attr;
  onnx_operator run;
} operator__onnx__XXX__context;*/


#endif

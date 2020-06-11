
//this file was generated by ../../../scripts/onnx_generator/OperatorSanityCheck.py
#include "operators/check_operator.h"
#include "operators/onnx/operator__onnx__maxpool__12.h"

bool check_operator__onnx__maxpool__12(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    size_t                  n_attribute,
    Onnx__AttributeProto ** attribute,
    size_t                  n_output,
    Onnx__TensorProto    ** output
){
    bool valid = true;
    { // check if input tensors have valid types
        uint32_t types_X[5] = {
            ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT8,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT8
        };
        check_operator_condition_tensor conditions[1] = {
            {
                .skip = false,
                .name = "X",
                .optional = false,
                .n_types = 5,
                .types  = types_X
            }
        };
        valid &= check_operator_tensors("operator__onnx__maxpool__12 input",
                                         1,
                                         conditions,
                                         input);
    }
    { // check if output tensors have valid types
        uint32_t types_Y[5] = {
            ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT8,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT8
        };
        uint32_t types_Indices[1] = {
            ONNX__TENSOR_PROTO__DATA_TYPE__INT64
        };
        check_operator_condition_tensor conditions[2] = {
            {
                .skip = false,
                .name = "Y",
                .optional = false,
                .n_types = 5,
                .types  = types_Y
            },{
                .skip = false,
                .name = "Indices",
                .optional = true,
                .n_types = 1,
                .types  = types_Indices
            }
        };
        valid &= check_operator_tensors("operator__onnx__maxpool__12 output",
                                         2,
                                         conditions,
                                         output);
    }
    { // check if attributes have valid types
        check_operator_condition_attribute conditions[7] = {
            {
                .skip = false,
                .name = "auto_pad",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING,
            },{
                .skip = false,
                .name = "ceil_mode",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT,
            },{
                .skip = false,
                .name = "dilations",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS,
            },{
                .skip = false,
                .name = "kernel_shape",
                .optional = true,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS,
            },{
                .skip = false,
                .name = "pads",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS,
            },{
                .skip = false,
                .name = "storage_order",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT,
            },{
                .skip = false,
                .name = "strides",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS,
            }
        };
        valid &= check_operator_attributes("operator__onnx__maxpool__12",
                                           7,
                                           conditions,
                                           attribute);
    }
    { // check if multiple tensors constrained by 'T' have same type
        check_operator_condition_constraint conditions_input[1] = {
            {
                .skip = false,
                .name = "X",
                .optional = false
            }
        };
        check_operator_condition_constraint conditions_output[2] = {
            {
                .skip = false,
                .name = "Y",
                .optional = false
            },{
                .skip = true,
                .name = "Indices",
                .optional = true
            }
        };
        valid &= check_operator_constraint("operator__onnx__maxpool__12 T",
                                           1,
                                           conditions_input,
                                           input,
                                           2,
                                           conditions_output,
                                           output);
    }
    
    
    { // check if multiple tensors constrained by 'I' have same type
        check_operator_condition_constraint conditions_input[1] = {
            {
                .skip = true,
                .name = "X",
                .optional = false
            }
        };
        check_operator_condition_constraint conditions_output[2] = {
            {
                .skip = true,
                .name = "Y",
                .optional = false
            },{
                .skip = false,
                .name = "Indices",
                .optional = true
            }
        };
        valid &= check_operator_constraint("operator__onnx__maxpool__12 I",
                                           1,
                                           conditions_input,
                                           input,
                                           2,
                                           conditions_output,
                                           output);
    }
    return valid;
}

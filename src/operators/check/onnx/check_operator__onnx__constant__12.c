
//this file was generated by ../../../scripts/onnx_generator/OperatorSanityCheck.py
#include "operators/check_operator.h"
#include "operators/onnx/operator__onnx__constant__12.h"

bool check_operator__onnx__constant__12(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    size_t                  n_attribute,
    Onnx__AttributeProto ** attribute,
    size_t                  n_output,
    Onnx__TensorProto    ** output
){
    bool valid = true;
    { // check if input tensors have valid types
        
        check_operator_condition_tensor conditions[0] = {
            
        };
        valid &= check_operator_tensors("operator__onnx__constant__12 input",
                                         0,
                                         conditions,
                                         input);
    }
    { // check if output tensors have valid types
        uint32_t types_output[15] = {
            ONNX__TENSOR_PROTO__DATA_TYPE__BOOL,
            ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128,
            ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64,
            ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
            ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT16,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT64,
            ONNX__TENSOR_PROTO__DATA_TYPE__INT8,
            ONNX__TENSOR_PROTO__DATA_TYPE__STRING,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT16,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT32,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT64,
            ONNX__TENSOR_PROTO__DATA_TYPE__UINT8
        };
        check_operator_condition_tensor conditions[1] = {
            {
                .skip = false,
                .name = "output",
                .optional = false,
                .n_types = 15,
                .types  = types_output
            }
        };
        valid &= check_operator_tensors("operator__onnx__constant__12 output",
                                         1,
                                         conditions,
                                         output);
    }
    { // check if attributes have valid types
        check_operator_condition_attribute conditions[8] = {
            {
                .skip = false,
                .name = "sparse_value",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR,
            },{
                .skip = false,
                .name = "value",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR,
            },{
                .skip = false,
                .name = "value_float",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT,
            },{
                .skip = false,
                .name = "value_floats",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS,
            },{
                .skip = false,
                .name = "value_int",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT,
            },{
                .skip = false,
                .name = "value_ints",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS,
            },{
                .skip = false,
                .name = "value_string",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING,
            },{
                .skip = false,
                .name = "value_strings",
                .optional = false,
                .type = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS,
            }
        };
        valid &= check_operator_attributes("operator__onnx__constant__12",
                                           8,
                                           conditions,
                                           attribute);
    }
    { // check if multiple tensors constrained by 'T' have same type
        check_operator_condition_constraint conditions_input[0] = {
            
        };
        check_operator_condition_constraint conditions_output[1] = {
            {
                .skip = false,
                .name = "output",
                .optional = false
            }
        };
        valid &= check_operator_constraint("operator__onnx__constant__12 T",
                                           0,
                                           conditions_input,
                                           input,
                                           1,
                                           conditions_output,
                                           output);
    }
    return valid;
}

import inspect
import os
import pathlib
import re
import itertools

from .Template import Template

class Range(Template):
    _template = '''
{{ {min}, {max} }}
'''
    def __init__(self, _range):
        self._range = _range
        self.min = _range[0]
        self.max = _range[1]


class Attribute(Template):
    _template = '''
{{
    .name     = "{name}",
    .optional = {optional},
    .type     = {type}
}}
'''
    def __init__(self, attribute):
        self.attribute = attribute
        self.name = attribute.name
        self.optional = "true" if attribute.optional else "false"
        self.type = attribute.onnxAttributeDataType()

class AttributeList(Template):
    _template = '''
static
operator_info_attribute
{cname}[] = {{
{attributes}
}};
'''
    def __init__(self, schema):
        self.schema = schema
        self.cname = "attributes"
        self.domain = schema.domain
        self.operator_name = schema.operator_name
        self._attributes = [Attribute(a) for a in schema.attributes]
        self.attributes = ",\n".join((str(x) for x in self._attributes))

    def __len__(self):
        return len(self._attributes)

class TypeList(Template):
    _template = '''
static
uint32_t
{cname}[] = {{
{types}
}};
'''
    def __init__(self, tensor, prefix=""):
        self.tensor = tensor
        self.cname = prefix + "tensor_type_" + tensor.name
        self._types = [t.onnxTensorDataTypes() for t in tensor.types]
        self.types = ",\n".join(itertools.chain(*self._types))

    def __len__(self):
        return len(self._types)

class _Tensor(Template):
    _template = '''
{{
    .name        = "{name}",
    .optional    = {optional},
    .variadic    = {variadic},
    .homogeneous = {homogeneous},
    .constraint  = "{constraint}",
    .n_types     = {n_types},
    .types       = {cname_types}
}}
'''
    def __init__(self, tensor):
        self.tensor = tensor
        self.name = tensor.name
        self.optional = "true" if tensor.optional else "false"
        self.variadic = "true" if tensor.optional else "false"
        self.homogeneous = "true" if tensor.isHomogeneous else "false"
        self.constraint   = tensor.constraint
        # See TensorInput / TensorOutput
        # self._types = TypeList(tensor)
        # self.cname_types = self._types.cname
        # self.n_types = len(self._types._types)

class TensorInput(_Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)
        self._types = TypeList(tensor,"input_")
        self.cname_types = self._types.cname
        self.n_types = len(self._types._types)

class TensorOutput(_Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)
        self._types = TypeList(tensor,"output_")
        self.cname_types = self._types.cname
        self.n_types = len(self._types._types)

class InputList(Template):
    _template = '''
{types}

static
operator_info_tensor
{cname}[] = {{
{inputs}
}};
'''
    def __init__(self, schema):
        self.schema = schema
        self.cname = "inputs"
        self.domain = schema.domain
        self.operator_name = schema.operator_name
        self._inputs = [TensorInput(t) for t in schema.inputs]
        self.inputs  = ",\n".join((str(i) for i in self._inputs))
        self.types   = "\n\n".join((str(i._types) for i in self._inputs))

    def __len__(self):
        return len(self._inputs)

class OutputList(Template):
    _template = '''
{types}

static
operator_info_tensor
{cname}[] = {{
{outputs}
}};
'''
    def __init__(self, schema):
        self.schema = schema
        self.cname = "outputs"
        self.domain = schema.domain
        self.operator_name = schema.operator_name
        self._outputs = [TensorOutput(t) for t in schema.outputs]
        self.outputs  = ",\n".join((str(i) for i in self._outputs))
        self.types   = "\n\n".join((str(i._types) for i in self._outputs))

    def __len__(self):
        return len(self._outputs)

class Constraint(Template):
    _template = '''
{{ "{name}" }}
'''
    def __init__(self, constraint):
        self.constraint = constraint
        self.name = constraint.name

class ConstraintList(Template):
    _template = '''
static
operator_info_constraint
{cname}[] = {{
{constraints}
}};
'''
    def __init__(self, schema):
        self.schema = schema
        self.cname = "constraints"
        self._constraints = [Constraint(c) for c in schema.constraints.values()]
        self.constraints = ",\n".join((str(c) for c in self._constraints))

    def __len__(self):
        return len(self._constraints)

class Info(Template):
    _template = '''
/* attributes */
{attributes}

/* input tensors */
{inputs}

/* output tensors */
{outputs}

/* constraints */
{constraints}

/* operator info */
operator_info
{cname} = {{
    .name         = "{name}",
    .range_input  = {range_input},
    .range_output = {range_output},
    .n_attribute  = {n_attribute},
    .attribute    = {cname_attributes},
    .n_input      = {n_input},
    .input        = {cname_inputs},
    .n_output     = {n_output},
    .output       = {cname_outputs},
    .n_constraint = {n_constraint},
    .constraint   = {cname_constraints}
}};
'''
    def __init__(self, schema):
        self.schema = schema
        self.attributes = AttributeList(schema)
        self.inputs = InputList(schema)
        self.outputs = OutputList(schema)
        self.constraints = ConstraintList(schema)
        self.cname = "info_" + schema.operator_name
        self.name = schema.name
        self.range_input = Range(schema.range_input)
        self.range_output = Range(schema.range_output)
        self.n_attribute = len(self.attributes)
        self.cname_attributes = self.attributes.cname
        self.n_input = len(self.inputs)
        self.cname_inputs = self.inputs.cname
        self.n_output = len(self.outputs)
        self.cname_outputs = self.outputs.cname
        self.n_constraint = len(self.constraints)
        self.cname_constraints = self.constraints.cname

class Source(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.name}/{schema.version}/info_{schema.operator_name}.c"
    _template = '''
//this file was generated by {scriptpath}
#include "operators/operator_info.h"
#include "{header_path}"

{info}

'''

    def __init__(self, header, path):
        self.schema = header.schema
        self.header = header
        self.header_path = header.filepath(False, False).parts[-1]
        self.path = path
        self.info = Info(self.schema)

    def filename(self):
        return self.filepath()

import inspect
import os
import pathlib

class OperatorCheckAttributes:
    _template_condition = '''
            {{
                .skip = {skip},
                .name = "{name}",
                .optional = {optional},
                .type = {type},
            }}
'''
    _template_check = '''
    {{ // check if attributes have valid types
        operator_check_condition_attribute conditions[{n_conditions}] = {{
            {conditions}
        }};
        check &= operator_check_attributes("{operator_name}",
                                           {n_conditions},
                                           conditions,
                                           attribute);
    }}
'''

    def __init__(self, schema):
        self.schema = schema

    def text(self,prefix=""):
        conditions = []
        for attribute in self.schema.attributes:
            conditions.append(self._template_condition.format(
                skip = "false",
                name = attribute.name,
                optional = attribute.optional,
                type = attribute.type
            ).strip())

        return self._template_check.format(
            n_conditions = len(conditions),
            conditions = f",".join(conditions),
            operator_name = self.schema.operator_name,
        ).strip()

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__name__}({self.schema.__repr__()})"


class OperatorCheckTensors:
    _template_types = '''
    uint32_t types_{name}[{n_types}] = {{
        {types}
    }};
'''
    _template_condition = '''
        {{
            .skip = {skip},
            .name = "{name}",
            .optional = {optional},
            .n_types = {n_types},
            .types  = types_{name}
        }}
'''
    _template_check = '''
{{ // check if {tensors} tensors have valid types
    {types}
    operator_check_condition_tensor conditions[{n_conditions}] = {{
        {conditions}
    }};
    valid &= operator_check_tensors("{operator_name}",
                                     {n_conditions},
                                     conditions,
                                     {tensors});
}}
'''
    def __init__(self, schema):
        self.schema = schema

    def text(self, prefix=""):
        checks = []
        input_types = []
        input_conditions = []
        output_types = []
        output_conditions = []
        for input in self.schema.inputs:
            types = [ t.onnxTensorDataTypes()[0] for t in input.types ]
            input_types.append(self._template_types.format(
                name = input.name,
                n_types = len(input.types),
                types = f",\n{' '*8}".join(types)
            ).strip())
            input_conditions.append(self._template_condition.format(
                skip = "false",
                name = input.name,
                optional = "true" if input.optional else "false",
                n_types = len(types)
            ).strip())
        checks.append(self._template_check.format(
            types = f"\n{' '*4}".join(input_types),
            n_conditions = len(input_conditions),
            conditions = ",".join(input_conditions),
            operator_name = f"{self.schema.operator_name} input",
            tensors = "input"
        ).strip())
        for output in self.schema.outputs:
            types = [ t.onnxTensorDataTypes()[0] for t in output.types ]
            output_types.append(self._template_types.format(
                name = output.name,
                n_types = len(output.types),
                types = f"\n{' '*8}".join(types)
            ).strip())
            output_conditions.append(self._template_condition.format(
                skip = "false",
                name = output.name,
                optional = "true" if output.optional else "false",
                n_types = len(types)
            ).strip())
        checks.append(self._template_check.format(
            types = f"\n{' '*4}".join(output_types),
            n_conditions = len(output_conditions),
            conditions = ",".join(output_conditions),
            operator_name = f"{self.schema.operator_name} output",
            tensors = "output"
        ).strip())
        return f"\n".join(checks).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__name__}({self.schema.__repr__()})"


class OperatorCheckRange:
    _template = '''
{{ // check if argument number is in valid range
    operator_check_condition_range condition_input = {{
        .name = "input",
        .min  = {min_input},
        .max  = {max_input}
    }};
    operator_check_condition_range condition_output = {{
        .name = "output",
        .min  = {min_output},
        .max  = {max_output}
    }};
    valid &= operator_check_constraint({operator_name},
                                       &condition_input,
                                       n_input);
    valid &= operator_check_constraint({operator_name},
                                       &condition_output,
                                       n_output);
}}
'''

    def __init__(self, schema):
        self.schema = schema

    def text(self, prefix=""):
        return self._template.format(
            max_input = self.schema.range_input[1],
            max_output = self.schema.range_output[1],
            min_input = self.schema.range_input[0],
            min_output = self.schema.range_output[0],
            operator_name = self.schema.operator_name,
        ).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__name__}({self.schema.__repr__()})"

class OperatorCheckConstraints:
    _template_condition = '''
        {{
            .skip = {skip},
            .name = "{name}",
            .optional = {optional}
        }}
'''
    _template_check = '''
{{ // check if multiple tensors constrained by '{constraint}' have same type
    operator_check_condition_constraint conditions_input[{n_conditions_input}] = {{
        {conditions_input}
    }};
    operator_check_condition_constraint conditions_output[{n_conditions_output}] = {{
        {conditions_output}
    }};
    valid &= operator_check_constraint("{operator_name} {constraint}",
                                       {n_conditions_input},
                                       conditions_input,
                                       input,
                                       {n_conditions_output},
                                       conditions_output,
                                       output);
}}
'''

    def __init__(self, schema):
        self.schema = schema

    def text(self,prefix=""):
        checks = []
        for constraint in self.schema.constraints.keys():
            conditions_input = []
            conditions_output = []
            for input in self.schema.inputs:
                conditions_input.append(self._template_condition.format(
                    skip = "false" if input.constraint == constraint else "true",
                    name = input.name,
                    optional = "true" if input.optional else "false",
                ).strip())
            for output in self.schema.outputs:
                conditions_output.append(self._template_condition.format(
                    skip = "false" if output.constraint == constraint else "true",
                    name = output.name,
                    optional = "true" if output.optional else "false",
                ).strip())

            checks.append(self._template_check.format(
                conditions_input = ",".join(conditions_input),
                conditions_output = ",".join(conditions_output),
                constraint = constraint,
                n_conditions_input = len(conditions_input),
                n_conditions_output = len(conditions_output),
                operator_name = self.schema.operator_name,
            ))
        return f"\n".join(checks).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__name__}({self.schema.__repr__()})"


class OperatorSanityCheck:
    _template = '''
//this file was generated by {script}
#include "{operator_name}.h"

bool {operator_name}_check(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    size_t                  n_attribute,
    Onnx__AttributeProto ** attribute,
    size_t                  n_output,
    Onnx__TensorProto    ** output
){{
    bool valid = true;
    {check_tensors}
    {check_attributes}
    {check_constraints}
    return valid;
}}
'''

    def __init__(self, schema, path):
        self.schema = schema
        self.path = path
        self.check_tensors = OperatorCheckTensors(self.schema).text(" "*4)
        self.check_attributes = OperatorCheckAttributes(self.schema).text(" "*4)
        self.check_constraints = OperatorCheckConstraints(self.schema).text(" "*4)

    def text(self):
      return self._template.format(
        script=self._rel_path(inspect.getfile(inspect.currentframe())),
        operator_name=self.schema.operator_name,
        check_tensors=self.check_tensors,
        check_attributes=self.check_attributes,
        check_constraints=self.check_constraints,
      )

    def filename(self):
        path = str(self.path)
        path += f"/{self.schema.domain}"
        path += f"/{self.schema.operator_name}_check.c"
        return pathlib.Path(path)

    def _rel_path(self, path):
        return os.path.relpath(os.path.realpath(path),os.path.realpath(self.path))

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__name__}({self.schema.__repr__()}, {self.path.__repr__()})"


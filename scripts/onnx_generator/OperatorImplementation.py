import inspect
import os

class OperatorTypeSwitch:
    _template_resolveType = '''
uint32_t {constraint} = 0;
{{ // get type of constraint '{constraint}'
    const char name = "{name}";
    Onnx__TensorProto *tensor = NULL;
    size_t n_tensor = operator_findTensors(&tensor, &name, 1, {inOrOutput}, n_{inOrOutput});
    if (n_tensor == 0) {{
        fprintf(stderr,"tensor '%s' not found!", name);
        exit(1);
    }}
    {constraint} = tensor->data_type;
}}
'''
    _template_switch = '''
switch ( {constraint} ) {{
    {cases}
    default: {{
        fprintf(stderr, "no matching type for constraint '{constraint}' found!\\n");
        exit(1);
    }}
}}
'''
    _template_case = '''
case {case}: {content}
'''

    _template = '''
{{
    {resolveTypes}
    {switch}
}}
'''
    def __init__(self, schema):
        self.schema = schema

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"OperatorTypeSwitch({self.schema.__repr__()})"

    def text(self, indent=4):
        resolveTypes = []
        cases = []

        for constraint in self.schema.constraints.keys():
            for input in self.schema.inputs:
                if constraint == input.constraint:
                    resolveTypes.append(self._template_resolveType.format(
                        constraint = constraint,
                        inOrOutput = "input",
                        name = input.name,
                    ).strip())
                    break
            else:
                for output in self.schema.outputs:
                    if constraint == output.constraint:
                        resolveTypes.append(self._template_resolveType.format(
                            constraint = constraint,
                            inOrOutput = "output",
                            name = output.name,
                        ).strip())
                        break

        permutationsMap = self.schema.constraints.typePermutationsMap()
        if not permutationsMap:
            return "/* skipping constraint test, because no constraint exist */"
        return self._template.format(
            resolveTypes = "\n".join(resolveTypes).strip().replace("\n",f"\n{' '*indent}"),
            switch = self._text_walkPermutationsMap(permutationsMap, indent).replace('\n','\n'+' '*indent)
        ).strip()

    def _text_walkPermutationsMap(self, node, indent=4):
        cases = []
        for k,v in node.items():
            case = k[-1][1].onnxTensorDataTypes()
            if not case:
                cases.append(f"/* skip non tensor constraint '{k[-1][0]}' ('{k[-1][1].original}') */")
                continue
            operator_name = self.schema.operator_name
            typePermutationText = self.schema.constraints.typePermutationText(k)
            if not v:
                cases.append(self._template_case.format(
                    case = case[0],
                    content = f"return &{operator_name}__{typePermutationText};"
                ).strip())
            else:
                cases.append(self._template_case.format(
                    case = case[0],
                    content = self._text_walkPermutationsMap(v,indent)
                ).strip())
        return self._template_switch.format(
            constraint = list(node.keys())[0][-1][0],
            cases = "\n".join(cases).replace('\n','\n'+' '*indent)
        ).strip()

class OperatorIsOneOfTypes:

    _template_check = '''
{{ // check if tensor '{name}' has valid type
    Onnx__TensorProto *tensor = NULL;
    const char *name = "{name}";
    size_t n_tensor = operator_findTensors(&tensor, &name, 1, {inOrOutput}, n_{inOrOutput});
    if (n_tensor == 0) {{
        fprintf(stderr,"tensor '%s' not found!", name);
        exit(1);
    }}
    uint32_t types[] = {{
        {types}
    }};
    if (!operator_tensorIsOneOfTypes(tensor, types, {n_types})) {{
        fprintf(stderr,
            "{inOrOutput} tensor '%s' has unexpected type: %u\\n",
            name,
            tensor->data_type
        );
        exit(1);
    }}
}}
'''
    def __init__(self, schema):
        self.schema = schema

    def text(self, prefix=""):
        checks = []
        for i in self.schema.inputs:
            types = set()
            for t in i.types:
                for tt in t.onnxTensorDataTypes():
                    types.add(tt)

            checks.append(self._template_check.format(
                inOrOutput = "input",
                n_types = len(types),
                name = i.name,
                types = ", ".join(types),
            ).strip())

        for o in self.schema.outputs:
            types = set()
            for t in o.types:
                for tt in t.onnxTensorDataTypes():
                    types.add(tt)
            checks.append(self._template_check.format(
                inOrOutput = "output",
                n_types = len(types),
                name = o.name,
                types = f",\n{prefix}{prefix}".join(types),
            ).strip())
        return f"\n".join(checks).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"OperatorIsOneOfTypes({self.schema.__repr__()})"


class OperatorCheckInOrOutputNumber:
    _template = '''
{{ // check if inputs and outputs have valid range
    if (n_input < {min_in}) {{
        fprintf(stderr,
            "mismatch of input tensors: "
            "found %" PRId64 ", "
            "expected at least {min_in}\\n",
            n_input
        );
        exit(1);
    }}
    if (n_input > {max_in}) {{
        fprintf(stderr,
            "mismatch of input tensors: "
            "found %" PRId64 ", "
            "expected at most {max_in}\\n",
            n_input
        );
        exit(1);
    }}
    if (n_output < {min_out}) {{
        fprintf(stderr,
            "mismatch of output tensors: "
            "found %" PRId64 ", "
            "expected at least {min_out}\\n",
            n_output
        );
        exit(1);
    }}
    if (n_output > {max_out}) {{
        fprintf(stderr,
            "mismatch of output tensors: "
            "found %" PRId64 ", "
            "expected at most {max_out}\\n",
            n_output
        );
        exit(1);
    }}
}}
'''

    def __init__(self, schema):
        self.schema = schema

    def text(self, prefix=""):
        return self._template.format(
            min_in = self.schema.range_input[0],
            max_in = self.schema.range_input[1],
            min_out = self.schema.range_output[0],
            max_out = self.schema.range_output[1],
        ).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"OperatorCheckInOrOutputNumber({self.schema.__repr__()})"

class OperatorCheckSameType:
    _template_check = '''
{{ // check if multiple tensors constrained by '{constraint}' have same type
    size_t n_tensors = 0;
    Onnx__TensorProto *tensors[{n_tensors_max}];
    {find_inputs}
    {find_outputs}
    if ( !operator_tensorsAreOfSameType( tensors, n_tensors ) ) {{
        fprintf(stderr, "tensor type mismatch between: ")
        for (size_t i = 0; i < n_tensors; i++) {{
            fprintf(stderr, "%s, ", tensors[i]->name);
        }}
    fprintf(stderr, "\\n");
    exit(1);
    }}
}}
'''
    _template_find = '''
{{ // find {inOrOutput}s for constraint '{constraint}'
    char *names = {{ {names} }};
    n_tensors += operator_findTensors(
        tensors,
        names,
        sizeof(names)/sizeof(*names),
        {inOrOutput_list},
        {inOrOutput_length}
    );
}}
'''

    def __init__(self, schema):
        self.schema = schema

    def text(self,prefix=""):
        checks = []
        constraint2both = {}
        constraint2input = {}
        constraint2output = {}
        for input in self.schema.inputs:
            if input.constraint in self.schema.constraints:
                constraint2both.setdefault(input.constraint, []).append(input)
                constraint2input.setdefault(input.constraint, []).append(input)
                constraint2output.setdefault(input.constraint, [])
        for output in self.schema.outputs:
            if output.constraint in self.schema.constraints:
                constraint2both.setdefault(output.constraint, []).append(output)
                constraint2input.setdefault(output.constraint, [])
                constraint2output.setdefault(output.constraint, []).append(output)
        for constraint, inOrOutputs in constraint2both.items():
            if len(inOrOutputs) < 2:
                checks.append(f"// skip check if multiple tensors constrained by '{constraint}' have same type ({len(inOrOutputs)} tensor)")
            else:
                find_inputs = f"// no input for constraint '{constraint}'"
                find_outputs = f"// no output for constraint '{constraint}'"
                inputs = constraint2input[constraint]
                outputs = constraint2output[constraint]
                if inputs:
                    find_inputs = self._template_find.format(
                        constraint=constraint,
                        inOrOutput="input",
                        names=" ".join([f'"{i.name}"' for i in inputs]),
                        inOrOutput_list="input",
                        inOrOutput_length=len(inputs)
                    ).strip()
                if outputs:
                    find_outputs = self._template_find.format(
                        constraint=constraint,
                        inOrOutput="output",
                        names=", ".join([f'"{o.name}"' for o in outputs]),
                        inOrOutput_list="output",
                        inOrOutput_length=len(outputs)
                    ).strip()
                checks.append(self._template_check.format(
                    constraint=constraint,
                    n_tensors_max=len(inOrOutputs),
                    find_inputs=find_inputs,
                    find_outputs=find_outputs
                ).strip())
        return f"\n".join(checks).strip().replace("\n",f"\n{prefix}")

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"OperatorCheckSameType({self.schema.__repr__()})"


class OperatorImplementation:
    _template = '''
//this file was generated by {script}
#include "{operator_name}.h"
#include <inttypes.h>
#include <stdio.h>

int {operator_name}(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    size_t                  n_attribute,
    Onnx__AttributeProto ** attribute,
    size_t                  n_output,
    Onnx__TensorProto    ** output
) {{
    return {operator_name}_resolve(
        n_input,
        input,
        n_attribute,
        attribute,
        n_output,
        output
    )(
        n_input,
        input,
        n_attribute,
        attribute,
        n_output,
        output
    );
}}

onnx_operator {operator_name}_resolve(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    size_t                  n_attribute,
    Onnx__AttributeProto ** attribute,
    size_t                  n_output,
    Onnx__TensorProto    ** output
){{
    {checks_inOrOutput_number}
    {checks_sameType}
    {checks_ofType}
    {switch}
}}
'''

    def __init__(self, schema, path):
        self.schema = schema
        self.path = path
        self.checks_inOrOutput_number = OperatorCheckInOrOutputNumber(self.schema).text(" "*4)
        self.checks_sameType = OperatorCheckSameType(self.schema).text(" "*4)
        self.checks_ofType = OperatorIsOneOfTypes(self.schema).text(" "*4)
        self.switch = OperatorTypeSwitch(self.schema).text()

    def text(self):
      return self._template.format(
        script=self._rel_path(inspect.getfile(inspect.currentframe())),
        operator_name=self.schema.operator_name,
        checks_inOrOutput_number=self.checks_inOrOutput_number,
        checks_sameType=self.checks_sameType,
        checks_ofType=self.checks_ofType,
        switch=self.switch
      )

    def _rel_path(self, path):
        return os.path.relpath(os.path.realpath(path),os.path.realpath(self.path))

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"OperatorImplementation({self.schema.__repr__()}, {self.path.__repr__()})"

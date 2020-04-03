import inspect
import itertools
import os
import sys
import re


def text_range_input(schema):
    return text_range(schema.min_input, schema.max_input)


def text_range_output(schema):
    return text_range(schema.min_output, schema.max_output)


def text_range(min, max):
    if (min == max):
        return f"always {min}"
    else:
        return f"{min} to {max}"


def format_text(prefix, start, texts):
    output = []
    curr = [start]
    linebreaks = 0
    for text in texts:
        lines = []
        length = len(prefix) + len(start)
        # split text into words by splitting on space and remove empty splits ("  ")
        # then split on newline boundaries, but keep emtpy splits ("\n\n")
        words = [w.split("\n") for w in text.strip().split(" ") if w != ""]
        words = list(itertools.chain(*words))
        for w in words:
            if w.strip() == "":
                # empty split, caused by "\n\n", should cause single line break
                if linebreaks >= 2:
                    # we already did 2 line breaks, skip this one
                    continue
                linebreaks += 1
                length = len(prefix) + len(start)
                lines.append(prefix + " ".join(curr))
                curr = [" "*len(start)]
                continue
            else:
                linebreaks = 0
            if length + len(w) < 79:
                # keep adding words
                length += len(w) + 1
                curr.append(w)
                continue

            # line is full, do line break
            length = len(prefix) + len(start) + len(w)
            lines.append(prefix + " ".join(curr))
            curr = [" "*len(start)]
            curr.append(w)
        lines.append(prefix + " ".join(curr))
        curr = [" "*len(start)]
        output.append("\n".join(lines))

    return "\n".join(output)


def sanitize_name(t):
    # order matters!
    replacements = {
        " ": "",
        "_": "_",
        ",": "_",
        "(": "",
        ")": "",
        "tensor": "t",
        "map": "m",
        "seq": "s",
        ".": "_"
    }
    for r in replacements.items():
        t = t.lower().replace(*r)
    return t.upper()


def permute_types(types):
    p = ['']
    for t in types:
        names = [
            f"{t.type_param_str}_{sanitize_name(n)}" for n in t.allowed_type_strs]
        p = ["__".join(x).strip("_") for x in itertools.product(p, names)]
    p.sort()
    return p


def text_inputs(schema):
    inputs = []
    for i in schema.inputs:
        allowed = [sanitize_name(t) for t in i.types]
        allowed.sort()
        allowed = ", ".join(allowed)
        type = sanitize_name(i.typeStr)
        inputs.append(f" * Input {type} {i.name}:")
        inputs.append(format_text(" * ", "  ", [i.description]))
        inputs.append(format_text(" *    ", "Allowed Types:", [allowed, '']))
    return "\n".join(inputs)


def text_outputs(schema):
    outputs = []
    for o in schema.outputs:
        allowed = [sanitize_name(t) for t in o.types]
        allowed.sort()
        allowed = ", ".join(allowed)
        type = sanitize_name(o.typeStr)
        outputs.append(f" * Output {type} {o.name}:")
        outputs.append(format_text(" * ", "  ", [o.description]))
        outputs.append(format_text(" *    ", "Allowed Types:", [allowed, '']))
    return "\n".join(outputs)


def text_constraints(schema):
    constraints = []
    for c in schema.type_constraints:
        allowed = [sanitize_name(t) for t in c.allowed_type_strs]
        allowed.sort()
        allowed = ", ".join(allowed)
        type = sanitize_name(c.type_param_str)
        constraints.append(f" * Contraint {type}:")
        constraints.append(format_text(" * ", "  ", [c.description]))
        constraints.append(format_text(
            " *    ", "Allowed Types:", [allowed, '']))
    return"\n".join(constraints)


def text_attributes(schema):
    attributes = []
    for a in schema.attributes.values():
        required = "(optional)"
        if a.required:
            required = "(required)"
        type = sanitize_name(a.type.name)
        attributes.append(f" * Attribute {type} {a.name} {required}:")
        attributes.append(format_text(" *", "  ", [a.description, '']))
    return "\n".join(attributes)


def text_deprecated(schema):
    return " * " + "@deprecated Avoid usage!" * schema.deprecated


def text_attribute_deprecated(schema):
    return "__attribute__((deprecated))\n" * schema.deprecated


def text_domain_name(schema):
    return sanitize_name(schema.domain + "_" + schema.name)


def text_doc_ref(schema):
    if schema.domain == '':
        return f" * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#{schema.name}\n"
    elif schema.domain == 'ai.onnx.ml':
        return f" * @see https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#{schema.name}\n"
    else:
        return ''


def text_stubs(schema):
    stubs = []
    template_stub = '''
__attribute__((weak, alias("operator_stub")))
extern int operator_{nameLower}_{version}_{type}(
  size_t                  n_input,
  Onnx__TensorProto    ** input,
  size_t                  n_attribute,
  Onnx__AttributeProto ** attribute,
  size_t                  n_output,
  Onnx__TensorProto    ** output
);
'''
    for t in permute_types(schema.type_constraints):
        stubs.append(
            template_stub.format(
                nameLower=schema.name.lower(),
                version=schema.since_version,
                type=t
            )
        )
    return "".join(stubs)


def text_doc(schema):
    return format_text(" *", "", [schema.doc])


def text_definition(schema):
    path = os.path.relpath(schema.file, path_include)
    return f" * @see {path}:{schema.line}"


def text_version(schema):
    return f" * @since version {schema.since_version}"


def text_header_operator(schema):
    template_header = '''
//this file was generated by {script}
#ifndef OPERATOR_{nameUpper}_{version}_H
#define OPERATOR_{nameUpper}_{version}_H

#include "operator.h"
#include "onnx.pb-c.h"

/**
 * Onnx operator '{nameOrig}' version {version}
 *
 * @param[in]  n_input     Number of inputs ({range_input})
 * @param[in]  input       Array of pointers to the inputs
 * @param[in]  n_attribute Number of attributes
 * @param[in]  attribute   Array of pointers to the attributes
 * @param[in]  n_output    Numper of outputs ({range_output})
 * @param[out] output      Array of pointer to the outputs
 * @return                 Error code
 *
 * @retval     0        No Error
 * @retval     ENOSYS   Operator is stubbed
 * @retval     EINVAL   Invalid argument
 * @retval     ENOMEM   Out of Memory
 * @retval     EFAULT   Invalid addr
 * @retval     EDOM     Math argument out of domain
 * @retval     ERANGE   Math result not representable
 *
{doc}
{deprecated}
{constraints}
{inputs}
{outputs}
{attributes}
{text_version}
{definition}
{documentation}
 */
{deprecated_attribute} int operator_{nameLower}_{version}(
  size_t                  n_input,
  Onnx__TensorProto    ** input,
  size_t                  n_attribute,
  Onnx__AttributeProto ** attribute,
  size_t                  n_output,
  Onnx__TensorProto    ** output
);
{stubs}
#endif
'''
    return template_header.format(
        attributes=text_attributes(schema),
        deprecated_attribute=text_attribute_deprecated(schema),
        deprecated=text_deprecated(schema),
        doc=text_doc(schema),
        documentation=text_doc_ref(schema),
        definition=text_definition(schema),
        range_input=text_range_input(schema),
        inputs=text_inputs(schema),
        line=schema.line,
        nameOrig=schema.name,
        nameLower=schema.name.lower(),
        nameUpper=schema.name.upper(),
        range_output=text_range_output(schema),
        outputs=text_outputs(schema),
        script=text_scriptname(),
        stubs=text_stubs(schema),
        version=schema.since_version,
        text_version=text_version(schema),
        constraints=text_constraints(schema),
    )


def text_scriptname():
    return inspect.getfile(inspect.currentframe())


def text_include_operator(schema):
    return f'#include "operator_{schema.name.lower()}.h"'


def text_header_operators(schemas):
    template_operators = '''
//this file was generated by {script}

#ifndef OPERATORS_H
#define OPERATORS_H

{includes}

#endif
'''
    includes = "\n".join([text_include_operator(s) for s in schemas])
    script = text_scriptname(),
    return template_operators.format(
        includes=includes,
        script=text_scriptname(),
    )


def map_types(schemas):
    typemap = {}
    for s in schemas:
        for c in s.type_constraints:
            for s in c.allowed_type_strs:
                name = sanitize_name(s)
                if s not in typemap:
                    typemap[s] = name
                else:
                    if typemap[s] != name:
                        raise "TYPE NAME SANITIZER PRODUCES CONFLICTS"
    return typemap


def text_check_sameType_inline_iterate_name_checks(inputOrOutput, names):
    template = '''
      if ( strcmp({inputOrOutput}[i]->name,"{name}") == 0 ) {{
        group[n_group++] = {inputOrOutput}[i];
        continue;
      }}
'''
    return " else ".join([
        template.format(
            inputOrOutput=inputOrOutput,
            name=name
        ).strip() for name in names
    ])


def text_check_sameType_inline_iterate(inputOrOutput, names):
    template = '''
    for (int i = 0; i < n_{inputOrOutput}; i++) {{
      {name_checks}
    }}
  '''
    return template.format(
        inputOrOutput=inputOrOutput,
        name_checks=text_check_sameType_inline_iterate_name_checks(
            inputOrOutput, names)
    ).strip()


def text_check_sameType_inline(constraint, input_names, output_names):
    template_check = '''
  // check if all inputs/outputs of shared constraint {constraint} have same type
  {{
    Onnx__TensorProto *group[{max_n_group}];
    size_t n_group = 0;
    {iterate_inputs}
    {iterate_outputs}
    for (int i = 1; i < n_group; i++) {{
      if ( group[0]->data_type != group[i]->data_type ) {{
        return EINVAL;
      }}
    }}
  }}
  '''
    template_skip = '''
  // skipping check for shared constraint {constraint} (single {inputOrOutput} {name})
  '''
    if len(input_names) + len(output_names) < 2:
        inputOrOutput = "FIXME"
        name = "FIXME"
        if len(input_names) == 1:
            inputOrOutput = "input"
            name = input_names[0]
        else:
            inputOrOutput = "output"
            name = output_names[0]
        return template_skip.format(
            constraint=constraint,
            inputOrOutput=inputOrOutput,
            name=name,
        ).strip()
    return template_check.format(
        constraint=constraint,
        max_n_group=len(input_names) + len(output_names),
        iterate_inputs=text_check_sameType_inline_iterate(
            "input", input_names),
        iterate_outputs=text_check_sameType_inline_iterate(
            "output", output_names),
    )


def text_check_sameType(schema):
    constraint2name = {}
    outputConstraint2name = {}
    inputConstraint2name = {}
    checks = []

    for i in schema.inputs:
        constraint2name.setdefault(i.typeStr, []).append(i.name)
        inputConstraint2name.setdefault(i.typeStr, []).append(i.name)
        outputConstraint2name.setdefault(i.typeStr, [])
    for o in schema.outputs:
        constraint2name.setdefault(o.typeStr, []).append(o.name)
        outputConstraint2name.setdefault(o.typeStr, []).append(o.name)
        inputConstraint2name.setdefault(o.typeStr, [])

    for constraint, names in constraint2name.items():
        checks.append(
            text_check_sameType_inline(
                constraint,
                inputConstraint2name[constraint],
                outputConstraint2name[constraint],
            )
        )
    return "\n".join(checks)


''' needs function like this
  static __attribute__((always_inline))
  bool operator_test_sameType(
    size_t                  n_input,
    Onnx__TensorProto    ** input,
    char                 ** input_names,
    size_t                  n_output,
    Onnx__TensorProto    ** output,
    char                 ** output_names,
    size_t                  max_n_group
  )
  {
    if ( max_n_group < 2 ) return TRUE;
    Onnx__TensorProto *group[max_n_group];
    size_t n_group = 0;
    for ( char **name = input_names; name; name++ ) {
      for (int i = 0; i < n_input; i++) {
        if ( strcmp(input[i]->name,*name) == 0 ) {
          group[n_group++] = input[i];
          break;
        }
      }
    }
    for ( char **name = output_names; name; name++ ) {
      for (int i = 0; i < n_output; i++) {
        if ( strcmp(output[i]->name,*name) == 0 ) {
          group[n_group++] = output[i];
          break;
        }
      }
    }
    for (int i = 1; i < n_group; i++) {
      if ( group[0]->data_type != group[i]->data_type ) {
        return FALSE;
      }
    }
    return TRUE;
  }
'''


def text_check_sameType_func_call(constraint, inputs, outputs):
    template = '''
  // check if all inputs/outputs of shared constraint {constraint} have same type
  {{
    char *input_names[] = {{ {input_names}, NULL }};
    char *output_names[] = {{ {output_names}, NULL }};
    if (!{function}(n_input, input, input_names, n_output, output, output_names, {max_n_group})) {{
      return EINVAL;
    }}
  }}
  '''
    return template.format(
        constraint=constraint,
        function="operator_test_sameType",
        input_names=" ,".join([f'"{name}"' for name in inputs]),
        output_names=" ,".join([f'"{name}"' for name in outputs]),
        max_n_group=len(inputs) + len(outputs),
    )


def text_check_sameType_func(schema):
    constraint2name = {}
    outputConstraint2name = {}
    inputConstraint2name = {}
    checks = []

    for i in schema.inputs:
        constraint2name.setdefault(i.typeStr, []).append(i.name)
        inputConstraint2name.setdefault(i.typeStr, []).append(i.name)
        outputConstraint2name.setdefault(i.typeStr, [])
    for o in schema.outputs:
        constraint2name.setdefault(o.typeStr, []).append(o.name)
        outputConstraint2name.setdefault(o.typeStr, []).append(o.name)
        inputConstraint2name.setdefault(o.typeStr, [])

    for constraint, names in constraint2name.items():
        checks.append(
            text_check_sameType_func_call(
                constraint,
                inputConstraint2name[constraint],
                outputConstraint2name[constraint],
            )
        )
    return "\n".join(checks)


def text_check_isType_func_call(inputOrOutput, name, types):
    template = '''
  {{
    Onnx__TensorProto__DataType types[] = {{ {types} , NULL}};
    if (!{function}({inputOrOutput}, {name}, types)) {{
      return EINVAL;
    }}
  }}
  '''
    return template.format(
        types=[parse_valueType(t) for t in types],
        inputOrOutput=inputOrOutput,
        function="operator_test_isType",
        name=f'"{name}"',
    )


def text_check_isType_func(schema):
    checks = []

    for i in schema.inputs:
        checks.append(
            text_check_isType_func_call(
                "input",
                i.name,
                i.types
            )
        )
    for o in schema.outputs:
        checks.append(
            text_check_isType_func_call(
                "output",
                o.name,
                o.types
            )
        )
    return "\n".join(checks)


def tokenize_valueType(typeStr):
    tokens = [
        "tensor",
        "map",
        "seq",
        "\\(",
        "\\)",
        "float",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "int32",
        "int64",
        "string",
        "bool",
        "float16",
        "double",
        "uint32",
        "uint64",
        "complex64",
        "complex128",
        "bfloat16",
        ",",
    ]
    re_scanner = re.compile("|".join([f" *({t})" for t in tokens]))
    return list(filter(None, re_scanner.split(typeStr)))


def parse_valueType(typeStr):
    tokens = tokenize_valueType(typeStr)

    def rule_start():
        result = rules[tokens.pop(0)]()
        if not tokens:
            return result
        print(tokens)
        raise "parser error in rule_start"

    def rule_tensor():
        if tokens.pop(0) == '(':
            result = rules[tokens.pop(0)]()
            if tokens.pop(0) == ')':
                return {"tensor": result}
        raise "parser error in rule_tensor"

    def rule_map():
        if tokens.pop(0) == '(':
            key = rules[tokens.pop(0)]()
            if tokens.pop(0) == ',':
                value = rules[tokens.pop(0)]()
                if tokens.pop(0) == ')':
                    return {"map": {"key": key, "value": value}}
        raise "parser error in rule_map"

    def rule_seq():
        if tokens.pop(0) == '(':
            result = rules[tokens.pop(0)]()
            if tokens.pop(0) == ')':
                return {"seq": result}
        raise "parser error in rule_seq"

    rules = {
        "tensor":     rule_tensor,
        "map":        rule_map,
        "seq":        rule_seq,
        "float":      lambda: "float",
        "uint8":      lambda: "uint8",
        "int8":       lambda: "int8",
        "uint16":     lambda: "uint16",
        "int16":      lambda: "int16",
        "int32":      lambda: "int32",
        "int64":      lambda: "int64",
        "string":     lambda: "string",
        "bool":       lambda: "bool",
        "float16":    lambda: "float16",
        "double":     lambda: "double",
        "uint32":     lambda: "uint32",
        "uint64":     lambda: "uint64",
        "complex64":  lambda: "complex64",
        "complex128": lambda: "complex128",
        "bfloat16":   lambda: "bfloat16",
    }

    return rule_start()


def valueType2enum(typeStr):
    dataTypes = {
        "tensor":     "ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE",
        "seq":        "ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE",
        "map":        "ONNX__TYPE_PROTO__VALUE_MAP_TYPE",
        "float":      "ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT",
        "uint8":      "ONNX__TENSOR_PROTO__DATA_TYPE__UINT8",
        "int8":       "ONNX__TENSOR_PROTO__DATA_TYPE__INT8",
        "uint16":     "ONNX__TENSOR_PROTO__DATA_TYPE__UINT16",
        "int16":      "ONNX__TENSOR_PROTO__DATA_TYPE__INT16",
        "int32":      "ONNX__TENSOR_PROTO__DATA_TYPE__INT32",
        "int64":      "ONNX__TENSOR_PROTO__DATA_TYPE__INT64",
        "string":     "ONNX__TENSOR_PROTO__DATA_TYPE__STRING",
        "bool":       "ONNX__TENSOR_PROTO__DATA_TYPE__BOOL",
        "float16":    "ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16",
        "double":     "ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE",
        "uint32":     "ONNX__TENSOR_PROTO__DATA_TYPE__UINT32",
        "uint64":     "ONNX__TENSOR_PROTO__DATA_TYPE__UINT64",
        "complex64":  "ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64",
        "complex128": "ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128",
        "bfloat16":   "ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16",
    }
    return dataTypes[typeStr]


def generate_headers_operator(path, schemas):
    for schema in schemas:
        path_dir = f"{path}/{schema.domain}"
        path_file = f"{path_dir}/operator_{schema.name.lower()}.h"
        text = text_header_operator(schema)
        os.makedirs(path_dir, exist_ok=True)
        open(path_file, "w").write(text)


def generate_header_operators(path, schemas):
    path_file = f"{path}/operators.h"
    text = text_header_operators(schemas)
    os.makedirs(path, exist_ok=True)
    open(path_file, "w").write(text)


def onnx_cpp2py_export_schemas(path):
    sys.path.insert(0, path)
    import onnx_cpp2py_export
    return onnx_cpp2py_export.defs.get_all_schemas()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "onnx_cpp2py_export",
        help="path to directory of onnx_cpp2py_export.so"
    )
    parser.add_argument(
        "include",
        help="path to directory where to save headers"
    )
    args = parser.parse_args()

    path_onnx_cpp2py_export = os.path.realpath(
        os.path.abspath(args.onnx_cpp2py_export))
    path_include = os.path.realpath(os.path.abspath(args.include))

    print(f"path onnx_cpp2py_export: {path_onnx_cpp2py_export}")
    print(f"path include: {path_include}")

    schemas = onnx_cpp2py_export_schemas(path_onnx_cpp2py_export)

    generate_header_operators(path_include, schemas)
    generate_headers_operator(path_include, schemas)

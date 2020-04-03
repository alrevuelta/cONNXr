import inspect
import itertools
import os
import sys


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

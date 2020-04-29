import os
import inspect
import pathlib

class OperatorHeaderContextArray:
    _template = '''
typedef struct operator_context_{kind}__{name} {{
    size_t length;
    {content}
}} operator_context_{kind}_{name};
'''

    def __init__(self, name, kind, content):
        self.name = name
        self.kind = kind
        self.content = content

    def text(self):
        return self._template.format(
            name=self.name,
            kind=self.kind,
            content = "\n".join([ f"{c}" for c in self.content])
        ).strip()

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"

class OperatorHeaderContextInput(OperatorHeaderContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "input",
            [f"operator_tensor *{i.name};" for i in self.schema.inputs]
        )

class OperatorHeaderContextOutput(OperatorHeaderContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "output",
            [f"operator_tensor *{o.name};" for o in self.schema.outputs]
        )
class OperatorHeaderContextAttribute(OperatorHeaderContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "attribute",
            [f"Onnx__AttributeProto *{a.name};" for a in self.schema.attributes]
        )
class OperatorHeaderContext:
    _template = '''
typedef struct operator_context__{name} {{
    struct operator_context_input__{name}     *input;
    struct operator_context_output__{name}    *output;
    struct operator_context_attribute__{name} *attribute;
    operator_executer                          operator;
}} operator_context__{name};
'''
    def __init__(self, schema):
        self.schema = schema

    def text(self):
        return self._template.format(
            name=self.schema.operator_name,
        ).strip()

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"

class OperatorHeaderPrototype:
    _template = '''
{attribute}
{return_type} {prefix}{name}{suffix}(
    operator_context__{name} *ctx
);
'''
    def __init__(self, name, prefix="",suffix="", return_type="operator_status", attribute=""):
        self.prefix = prefix
        self.name = name
        self.suffix = suffix
        self.attribute = attribute
        self.return_type = return_type

    def text(self):
        return self._template.format(
            name=self.name,
            prefix=self.prefix,
            suffix=self.suffix,
            attribute=self.attribute,
            return_type = self.return_type
        ).strip()

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"

class OperatorHeaderPrototypeAliases:
    def __init__(self, schema):
        self.schema = schema

    def text(self):
        return "\n".join([OperatorHeaderPrototypeAlias(
                            self.schema.operator_name,
                            t
                          ).text()
                          for t in self.schema.constraints.typePermutations() ])

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"

class OperatorHeaderPrototypeAlias(OperatorHeaderPrototype):
    def __init__(self, name, type):
        super().__init__(f"{name}", suffix=f"__{type}", attribute='extern __attribute__((weak))')

class OperatorHeaderPrototypeResolver(OperatorHeaderPrototype):
    def __init__(self, name):
        super().__init__(name, prefix=f"resolve_", return_type='operator_executer')

class OperatorHeaderDoxygen:
    _template = '''

{context_input}

{context_output}

{context_attribute}

{context}

/**
 * {domain} operator '{name}' version {version}
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
{doc}
{deprecated}
{constraints}
{inputs}
{outputs}
{attributes}
 *
 * @since version {version}
 *
{defs_filepath}
{doc_ref}
 */
'''
    def __init__(self, schema, path):
        self.schema = schema
        self.path   = path

    def text(self):
        return self._template.format(
            context_input = OperatorHeaderContextInput(self.schema),
            context_output = OperatorHeaderContextOutput(self.schema),
            context_attribute = OperatorHeaderContextAttribute(self.schema),
            context = OperatorHeaderContext(self.schema),
            attributes=self.schema.attributes.text(" * "),
            deprecated=" * @deprecated Avoid usage!" if self.schema.deprecated else " * ",
            doc=self.schema.doc.text(" * "),
            doc_ref=f" * @see {self.schema.ref_doc}",
            domain=self.schema.domain,
            constraints=self.schema.constraints.text(" * "),
            range_input=self._range(*self.schema.range_input),
            inputs=self.schema.inputs.text(" * "),
            name=self.schema.name,
            range_output=self._range(*self.schema.range_output),
            outputs=self.schema.outputs.text(" * "),
            version=self.schema.version,
            defs_filepath=f" * @see {self._rel_path(self.schema.ref_file[0])}:{self.schema.ref_file[1]}",
        ).strip()

    def _range(self, min, max):
        if (min == max):
            return f"exactly {min}"
        else:
            return f"{min} to {max}"

    def _rel_path(self, path):
        return os.path.relpath(os.path.realpath(path),os.path.realpath(self.path))

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"

class OperatorHeader:
    _template_header = '''
//this file was generated by {script}
# ifndef OPERATOR_{header_name}_H
# define OPERATOR_{header_name}_H

# include "operators/operator.h"
# include "operators/operator_stub.h"

{doxygen}
{prototype}

{resolver}

{aliases}
# endif
'''

    def __init__(self, schema, path):
      self.schema = schema
      self.path = path

    def text(self):
        return self._template_header.format(
            script=self._rel_path(inspect.getfile(inspect.currentframe())),
            header_name=self.schema.operator_name.upper(),
            operator_name=self.schema.operator_name,
            doxygen = OperatorHeaderDoxygen(self.schema, self.path),
            prototype = OperatorHeaderPrototype(self.schema.operator_name),
            resolver = OperatorHeaderPrototypeResolver(self.schema.operator_name),
            aliases = OperatorHeaderPrototypeAliases(self.schema)
        )

    def filename(self, path=None):
        path = str(self.path) if path == None else str(path)
        path += f"/{self.schema.domain}"
        path += f"/{self.schema.operator_name}.h"
        return pathlib.Path(path)

    def _rel_path(self, path):
        return os.path.relpath(os.path.realpath(path),os.path.realpath(self.path))

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"
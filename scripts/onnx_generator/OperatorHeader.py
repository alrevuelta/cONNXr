from .Template import Template

class ContextArray(Template):
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

class ContextInput(ContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "input",
            [f"operator_tensor *{i.name};" for i in self.schema.inputs]
        )

class ContextOutput(ContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "output",
            [f"operator_tensor *{o.name};" for o in self.schema.outputs]
        )
class ContextAttribute(ContextArray):
    def __init__(self, schema):
        self.schema = schema
        super().__init__(
            self.schema.operator_name,
            "attribute",
            [f"Onnx__AttributeProto *{a.name};" for a in self.schema.attributes]
        )
class Context(Template):
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
        self.name = schema.operator_name

class Prototype(Template):
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

class PrototypeAliases(Template):
    _template = "{aliases}"
    def __init__(self, schema):
        self.schema = schema
        self.aliases = "\n".join([str(PrototypeAlias(
                            self.schema.operator_name,
                            t
                          ))
                          for t in self.schema.constraints.typePermutations() ])

class PrototypeAlias(Prototype):
    def __init__(self, name, type):
        super().__init__(f"{name}", suffix=f"__{type}", attribute='extern __attribute__((weak))')

class PrototypeResolver(Prototype):
    def __init__(self, name):
        super().__init__(name, prefix=f"resolve_", return_type='operator_executer')

class Doxygen(Template):
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
        self.context_input = ContextInput(self.schema)
        self.context_output = ContextOutput(self.schema)
        self.context_attribute = ContextAttribute(self.schema)
        self.context = Context(self.schema)
        self.attributes=self.schema.attributes.text(" * ")
        self.deprecated=" * @deprecated Avoid usage!" if self.schema.deprecated else " * "
        self.doc=self.schema.doc.text(" * ")
        self.doc_ref=f" * @see {self.schema.ref_doc}"
        self.domain=self.schema.domain
        self.constraints=self.schema.constraints.text(" * ")
        self.range_input=self._range(*self.schema.range_input)
        self.inputs=self.schema.inputs.text(" * ")
        self.name=self.schema.name
        self.range_output=self._range(*self.schema.range_output)
        self.outputs=self.schema.outputs.text(" * ")
        self.version=self.schema.version
        self.defs_filepath=f" * @see {self.scriptpath(self.schema.ref_file[0])}:{self.schema.ref_file[1]}"

    def _range(self, min, max):
        if (min == max):
            return f"exactly {min}"
        else:
            return f"{min} to {max}"

class Info(Template):
    _template = '''
extern __attribute__((weak)) operator_info info_{schema.operator_name};
'''
    def __init__(self, schema):
        self.schema = schema

class Header(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.operator_name}.h"
    _template = '''
//this file was generated by {scriptpath}
# ifndef OPERATOR_{header_name}_H
# define OPERATOR_{header_name}_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

{doxygen}
{prototype}

{resolver}

{info}

{aliases}
# endif
'''

    def __init__(self, schema, path):
        self.schema = schema
        self.path = path
        self.header_name=schema.operator_name.upper()
        self.doxygen = Doxygen(schema, path)
        self.prototype = Prototype(schema.operator_name)
        self.resolver = PrototypeResolver(schema.operator_name)
        self.info = Info(schema)
        self.aliases = PrototypeAliases(schema)

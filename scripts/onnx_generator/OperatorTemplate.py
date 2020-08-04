from .Template import Template
import itertools

class Input(Template):
    _template = '''
// Onnx__TensorProto *i_{name} = searchInputByName(ctx, {index});
'''
    def __init__(self, input, index):
        self.input = input
        self.name = input.name
        self.index = index

class Output(Template):
    _template = '''
// Onnx__TensorProto *o_{name} = searchOutputByName(ctx, {index});
'''
    def __init__(self, output, index):
        self.output = output
        self.name = output.name
        self.index = index

class OutputFrees(Template):
    def __init__(self, output):
        self.output = output

    def __iter__(self):
        yield f"// freeTensorData(o_{self.output.name});"
        yield f"// free(o_{self.output.name}->dims);"

class Attribute(Template):
    _template = '''
// Onnx__AttributeProto *a_{name} = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"{name}");
'''
    def __init__(self, attr):
        self.attr = attr
        self.name = attr.name

class TraceInput(Template):
    _template = '''
// TRACE_TENSOR(2, {cond}, i_{name});
'''
    def __init__(self, input):
        self.input = input
        self.name = input.name
        self.cond = self.name if input.optional else "true"

class TraceOutput(Template):
    _template = '''
// TRACE_TENSOR(2, {cond}, o_{name});
'''
    def __init__(self, output):
        self.output = output
        self.name = output.name
        self.cond = self.name if output.optional else "true"

class TraceAttribute(Template):
    _template = '''
// TRACE_ATTRIBUTE(2, {cond}, a_{name});
'''
    def __init__(self, attr):
        self.attr = attr
        self.name = attr.name
        self.cond = f"a_{self.name}" if attr.optional else "true"

class TraceContext(Template):
    _template = '''
// {trace}
'''
    _trace = {
        "FLOAT"          : 'TRACE_VAR(2, true, {prefix}{name}, "%f");',
        "INT"            : 'TRACE_VAR(2, true, {prefix}{name}, "%" PRId64);',
        "STRING"         : 'TRACE_VAR(2, true, {prefix}{name}, "\\"%s\\"");',
        "TENSOR"         : 'TRACE_TENSOR(2, true, {prefix}{name});',
        "GRAPH"          : 'TRACE_GRAPH(2, true, {prefix}{name});',
        "SPARSE_TENSOR"  : 'TRACE_VAR(2, true, {prefix}{name}, "%p");',
        "FLOATS"         : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "%f");',
        "INTS"           : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "%" PRId64);',
        "STRINGS"        : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "\\"%s\\"");',
        "TENSORS"        : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "%p");',
        "GRAPHS"         : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "%p");',
        "SPARSE_TENSORS" : 'TRACE_ARRAY(2, true, {prefix}{name}, , {prefix}n_{name}, "%p");',
    }
    def __init__(self, attribute, prefix=""):
        self.attribute = attribute
        self.prefix = prefix
        self.trace = self._trace[attribute.type].format(name=attribute.name, prefix=prefix)

class ContextAssignments(Template):
    _assignments_optional = {
        "float"                     : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "int64_t"                   : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "size_t"                    : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "char*"                     : ('op_ctx->{n} = a_{a}?strndup((char*)a_{a}->{s}.data, a_{a}->{s}.len):default_{n};',),
        "Onnx__TensorProto*"        : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "Onnx__GraphProto*"         : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "Onnx__SparseTensorProto*"  : ('op_ctx->{n} = a_{a}?a_{a}->{s}:default_{n};',),
        "float*"                    : ('op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
        "int64_t*"                  : ('op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
        "char**"                    : ('if (a_{a}) {{',
                                       '    op_ctx->{n} = malloc(a_{a}->n_{s} * sizeof(char*));',
                                       '    TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");',
                                       '    for (int i = 0; i < a_{a}->n_{s}; i++) {{ op_ctx->{n}[i] = strndup((char*)a_{a}->{s}[i].data, a_{a}->{s}[i].len); }}',
                                       '}} else {{',
                                       '    op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       '    TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");',
                                       '}}'),
        "Onnx__TensorProto**"       : ('op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
        "Onnx__GraphProto**"        : ('op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
        "Onnx__SparseTensorProto**" : ('op_ctx->{n} = a_{a}?a_{a}->{s}:ARRAYDUP(default_{n},default_n_{n});',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
    }
    _assignments = {
        "float"                     : ('op_ctx->{n} = a_{a}->{s};',),
        "int64_t"                   : ('op_ctx->{n} = a_{a}->{s};',),
        "size_t"                    : ('op_ctx->{n} = a_{a}->{s};',),
        "char*"                     : ('op_ctx->{n} = strndup(a_{a}->{s}.data, a_{a}->{s}.len);',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");'),
        "Onnx__TensorProto*"        : ('op_ctx->{n} = a_{a}->{s};',),
        "Onnx__GraphProto*"         : ('op_ctx->{n} = a_{a}->{s};',),
        "Onnx__SparseTensorProto*"  : ('op_ctx->{n} = a_{a}->{s};',),
        "float*"                    : ('op_ctx->{n} = a_{a}->{s};',),
        "int64_t*"                  : ('op_ctx->{n} = a_{a}->{s};',),
        "char**"                    : ('op_ctx->{n} = malloc(a_{a}->n_{s} * sizeof(char*));',
                                       'TRACE_FATAL(0, !op_ctx->{n}, "malloc failed");',
                                       'for (int i = 0; i < a_{a}->n_{s}; i++) {{ op_ctx->{n}[i] = strndup(a_{a}->{s}[i].data, a_{a}->{s}[i].len); }}'),
        "Onnx__TensorProto**"       : ('op_ctx->{n} = a_{a}->{s};',),
        "Onnx__GraphProto**"        : ('op_ctx->{n} = a_{a}->{s};',),
        "Onnx__SparseTensorProto**" : ('op_ctx->{n} = a_{a}->{s};',),
    }

    def __init__(self, attribute):
        self.attribute = attribute

    def __iter__(self):
        for s,t,n in self.attribute.onnxAttributeDataTypeCDecl():
            if self.attribute.optional:
                for line in self._assignments_optional[t]:
                    yield "// " + line.format(a=self.attribute.name,n=n,t=t,s=s)
            else:
                for line in self._assignments[t]:
                    yield "// " + line.format(a=self.attribute.name,n=n,t=t,s=s)

class ContextFree(Template):
    _free = {
        "float"                     : (),
        "int64_t"                   : (),
        "size_t"                    : (),
        "char*"                     : ('free(op_ctx->{n});',),
        "Onnx__TensorProto*"        : ('free(op_ctx->{n});',),
        "Onnx__GraphProto*"         : ('free(op_ctx->{n});',),
        "Onnx__SparseTensorProto*"  : ('free(op_ctx->{n});',),
        "float*"                    : ('free(op_ctx->{n});',),
        "int64_t*"                  : ('free(op_ctx->{n});',),
        "char**"                    : ('for (int i = 0; i < op_ctx->n_{n}; i++) {{ free(op_ctx->{n}[i]); }}',
                                       'free(op_ctx->{n});'),
        "Onnx__TensorProto**"       : ('for (int i = 0; i < op_ctx->n_{n}; i++) {{ free(op_ctx->{n}[i]); }}',
                                       'free(op_ctx->{n});'),
        "Onnx__GraphProto**"        : ('for (int i = 0; i < op_ctx->n_{n}; i++) {{ free(op_ctx->{n}[i]); }}',
                                       'free(op_ctx->{n});'),
        "Onnx__SparseTensorProto**" : ('for (int i = 0; i < op_ctx->n_{n}; i++) {{ free(op_ctx->{n}[i]); }}',
                                       'free(op_ctx->{n});'),
    }

    def __init__(self, attribute):
        self.attribute = attribute

    def __iter__(self):
        for s,t,n in self.attribute.onnxAttributeDataTypeCDecl():
            for line in self._free[t]:
                yield "// " + line.format(a=self.attribute.name,n=n,t=t,s=s)

class ContextDefaults(Template):
    def __init__(self, attribute):
        self.attribute = attribute

    def __iter__(self):
        if self.attribute.optional:
            for s,t,n in self.attribute.onnxAttributeDataTypeCDecl():
                yield f"// {t} default_{n} = ;"

class ExecuteTemplate(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.name}/{schema.version}/execute_{schema.operator_name}{suffix}.c"
    _template = '''
//this file was generated by {scriptpath}
{include}
#include "tracing.h"
#include "utils.h"

operator_status
execute_{schema.operator_name}{suffix}(
    node_context *ctx
)
{{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    {inputs}

    {trace_inputs}

    {context}

    {declarations}

    {trace_context}

    {outputs}

    {trace_outputs}

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}}
'''
    def __init__(self, header, path, suffix=""):
        self.header = header
        self.path = path
        self.schema = header.schema
        self.suffix = suffix
        self.include = f'#include "{header.filepath().parts[-1]}"'
        self.inputs = "\n    ".join([str(Input(i,index)) for index,i in enumerate(self.schema.inputs)])
        self.outputs = "\n    ".join([str(Output(o,index)) for index,o in enumerate(self.schema.outputs)])
        self.attributes = "\n    ".join([str(Attribute(a)) for a in self.schema.attributes])
        self.trace_inputs = "\n    ".join([str(TraceInput(i)) for i in self.schema.inputs])
        self.trace_outputs = "\n    ".join([str(TraceOutput(o)) for o in self.schema.outputs])
        self.trace_attributes = "\n    ".join([str(TraceAttribute(a)) for a in self.schema.attributes])


        declarations = itertools.chain(*[a.onnxAttributeDataTypeCDecl() for a in self.schema.attributes])
        self.declarations = "\n    ".join(f"// {t} {n} = op_ctx->{n};" for s,t,n in declarations)

        self.context = ""
        if declarations:
            self.context = f"// context_{self.schema.operator_name} *op_ctx = ctx->executer_context;"

        self.trace_context = "\n    ".join([str(TraceContext(a)) for a in self.schema.attributes])

class PrepareTemplate(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.name}/{schema.version}/prepare_{schema.operator_name}.c"
    _template = '''
//this file was generated by {scriptpath}
{include}
#include "tracing.h"
#include "utils.h"

operator_status
prepare_{schema.operator_name}(
    node_context *ctx
)
{{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    {inputs}

    {trace_inputs}

    {attributes}

    {trace_attributes}

    {outputs}

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    {defaults}

    // context_{schema.operator_name} *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_{schema.operator_name}));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    {assignments}

    {trace_context}

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    {malloc_outputs}

    {trace_outputs}

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_{schema.operator_name}(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}}
'''
    def __init__(self, header, path):
        self.header = header
        self.path = path
        self.schema = header.schema
        self.include = f'#include "{header.filepath().parts[-1]}"'
        self.inputs = "\n    ".join([str(Input(i,index)) for index,i in enumerate(self.schema.inputs)])
        self.outputs = "\n    ".join([str(Output(o,index)) for index,o in enumerate(self.schema.outputs)])
        self.attributes = "\n    ".join([str(Attribute(a)) for a in self.schema.attributes])
        self.trace_inputs = "\n    ".join([str(TraceInput(i)) for i in self.schema.inputs])
        self.trace_outputs = "\n    ".join([str(TraceOutput(o)) for o in self.schema.outputs])
        self.trace_attributes = "\n    ".join([str(TraceAttribute(a)) for a in self.schema.attributes])

        assignments = itertools.chain(*[ContextAssignments(a) for a in self.schema.attributes])
        self.assignments = "\n    ".join([str(a) for a in assignments])

        defaults = itertools.chain(*[ContextDefaults(a) for a in self.schema.attributes])
        self.defaults = "\n    ".join([ str(d) for d in defaults])

        self.malloc_outputs = "\n    ".join([f"// mallocTensorData(o_{o.name});" for o in self.schema.outputs])
        self.trace_context = "\n    ".join([str(TraceContext(a,prefix="op_ctx->")) for a in self.schema.attributes])

class FreeTemplate(Template):
    _basepath = "{path}"
    _filepath = "{schema.domain}/{schema.name}/{schema.version}/free_{schema.operator_name}.c"
    _template = '''
//this file was generated by {scriptpath}
{include}
#include "tracing.h"
#include "utils.h"

void
free_{schema.operator_name}(
    node_context *ctx
)
{{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    {inputs}

    {trace_inputs}

    {attributes}

    {trace_attributes}

    {outputs}

    {trace_outputs}

    /* FREE CONTEXT HERE IF NEEDED */

    // context_{schema.operator_name} *op_ctx = ctx->executer_context;

    {trace_context}

    {free_context}

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    {free_outputs}

    TRACE_EXIT(1);
}}
'''
    def __init__(self, header, path):
        self.header = header
        self.path = path
        self.schema = header.schema
        self.include = f'#include "{header.filepath().parts[-1]}"'
        self.inputs = "\n    ".join([str(Input(i,index)) for index,i in enumerate(self.schema.inputs)])
        self.outputs = "\n    ".join([str(Output(o,index)) for index,o in enumerate(self.schema.outputs)])
        self.attributes = "\n    ".join([str(Attribute(a)) for a in self.schema.attributes])
        self.trace_inputs = "\n    ".join([str(TraceInput(i)) for i in self.schema.inputs])
        self.trace_outputs = "\n    ".join([str(TraceOutput(o)) for o in self.schema.outputs])
        self.trace_attributes = "\n    ".join([str(TraceAttribute(a)) for a in self.schema.attributes])
        self.trace_context = "\n    ".join([str(TraceContext(a,prefix="op_ctx->")) for a in self.schema.attributes])

        free_context = itertools.chain(*[ContextFree(a) for a in self.schema.attributes])
        self.free_context = "\n    ".join([str(f) for f in free_context])

        defaults = itertools.chain(*[ContextDefaults(a) for a in self.schema.attributes])
        self.defaults = "\n    ".join([ str(d) for d in defaults])

        free_outputs = itertools.chain(*[OutputFrees(o) for o in self.schema.outputs])
        self.free_outputs = "\n    ".join([str(f) for f in free_outputs])


class Templates:
    def __init__(self, header, path):
        self.header = header
        self.path = path

    def __iter__(self):
        yield PrepareTemplate(self.header, self.path)
        yield FreeTemplate(self.header, self.path)
        types = self.header.schema.constraints.typePermutations(filterInput=True)
        yield ExecuteTemplate(self.header, self.path)
        for t in types:
            yield ExecuteTemplate(self.header, self.path, suffix=f"__{t}")

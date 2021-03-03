import re
import itertools

def format_text(prefix, start, texts):
    output = []
    curr = []
    if start:
        curr.append(start)
    linebreaks = 0
    for text in texts:
        lines = []
        length = len(prefix)
        if start:
            length += + len(start)
        # split text into words by splitting on space and remove empty splits ("")
        # then split on newline boundaries, but keep empty splits ("\n\n")
        words = [w.split("\n") for w in text.strip().split(" ") if w != ""]
        words = list(itertools.chain(*words))
        for w in words:
            if w.strip() == "":
                if linebreaks == 0:
                    linebreaks += 1
                    continue
                if linebreaks >= 2:
                    # we already did 2 line breaks, skip this one
                    continue
                # empty split, caused by "\n\n", should cause single line break
                linebreaks += 1
                length = len(prefix)
                lines.append(prefix + " ".join(curr))
                curr = []
                if start:
                    length += len(start)
                    curr.append(" "*len(start))
                continue
            else:
                linebreaks = 0
            if length + len(w) < 79:
                # keep adding words
                length += len(w) + 1
                curr.append(w)
                continue

            # line is full, do line break
            length = len(prefix) + len(w)
            lines.append(prefix + " ".join(curr))
            curr = []
            if start:
                length += len(start)
                curr.append(" "*len(start))
            curr.append(w)
        lines.append(prefix + " ".join(curr))
        curr = []
        if start:
            curr.append(" "*len(start))
        output.append("\n".join(lines))

    return "\n".join(output)

class OnnxType(dict):
    _onnxTensorDataType = {
        "float": "ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT",
        "uint8": "ONNX__TENSOR_PROTO__DATA_TYPE__UINT8",
        "int8": "ONNX__TENSOR_PROTO__DATA_TYPE__INT8",
        "uint16": "ONNX__TENSOR_PROTO__DATA_TYPE__UINT16",
        "int16": "ONNX__TENSOR_PROTO__DATA_TYPE__INT16",
        "int32": "ONNX__TENSOR_PROTO__DATA_TYPE__INT32",
        "int64": "ONNX__TENSOR_PROTO__DATA_TYPE__INT64",
        "string": "ONNX__TENSOR_PROTO__DATA_TYPE__STRING",
        "bool": "ONNX__TENSOR_PROTO__DATA_TYPE__BOOL",
        "float16": "ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16",
        "double": "ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE",
        "uint32": "ONNX__TENSOR_PROTO__DATA_TYPE__UINT32",
        "uint64": "ONNX__TENSOR_PROTO__DATA_TYPE__UINT64",
        "complex64": "ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64",
        "complex128": "ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128",
        "bfloat16": "ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16",
    }

    class _Scanner:
        _tokens = {
            re.compile(r"tensor")     : "tensor"    ,
            re.compile(r"map")        : "map"       ,
            re.compile(r"seq")        : "seq"       ,
            re.compile(r"\(")         : "("         ,
            re.compile(r"\)")         : ")"         ,
            re.compile(r"float")      : "float"     ,
            re.compile(r"uint8")      : "uint8"     ,
            re.compile(r"int8")       : "int8"      ,
            re.compile(r"uint16")     : "uint16"    ,
            re.compile(r"int16")      : "int16"     ,
            re.compile(r"int32")      : "int32"     ,
            re.compile(r"int64")      : "int64"     ,
            re.compile(r"string")     : "string"    ,
            re.compile(r"bool")       : "bool"      ,
            re.compile(r"float16")    : "float16"   ,
            re.compile(r"double")     : "double"    ,
            re.compile(r"uint32")     : "uint32"    ,
            re.compile(r"uint64")     : "uint64"    ,
            re.compile(r"complex64")  : "complex64" ,
            re.compile(r"complex128") : "complex128",
            re.compile(r"bfloat16")   : "bfloat16"  ,
            re.compile(r",")          :  ","        ,
            re.compile(r"\s+")        :  None       ,
        }

        def __init__(self, string):
            self.string = string
            self.tokens = self.tokenize(string)

        def tokenize(self, string):
            pos = 0
            tokens = []
            while string[pos:]:
                allMatches = map(lambda x: (x[0].match(string[pos:]), x[1]), self._tokens.items())
                validMatches = filter(lambda x: x[0], allMatches)
                try:
                    longestMatch = max( validMatches, key=lambda x: x[0].end())
                except:
                    raise SyntaxError(f"no token matches: '{string[pos:]}'")
                else:
                    pos += longestMatch[0].end()
                    if longestMatch[1]:
                        tokens.append(longestMatch[1])
            return tokens

        def consume(self, expected_token = None):
            if not expected_token:
                return self.pop()
            if not self.peek(expected_token):
                raise SyntaxError(
                    f"expected '{expected_token}', but got '{self.peek()}'")
            return self.pop()

        def peek(self, expected_token=None):
            token = self.tokens[0]
            if expected_token:
                return token == expected_token
            else:
                return token

        def pop(self):
            return self.tokens.pop(0)

        def onToken(self, token2function, consume=False):
            for token, function in token2function.items():
                if self.peek(token):
                    if consume:
                        self.pop()
                    return function()
            tokens = ", ".join([f"'{t}'" for t in token2function.keys()])
            raise SyntaxError(f"expected one of {tokens}, but got '{self.peek()}'")

        def __repr__(self):
            return f"OnnxType._Scanner({self.string.__repr__()})"

    class _Parser:
        _terminals = {
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

        def __init__(self, scanner):
            self.scanner = scanner

        def __repr__(self):
            return f"OnnxType._Parser({self.scanner.__repr__()})"

        def _rule_tensor(self):
            self.scanner.consume('(')
            result = self.scanner.onToken(self._terminals, consume=True)
            self.scanner.consume(')')
            return {"tensor": result}

        def _rule_map(self):
            rules = {
                "tensor":     self._rule_tensor,
                "map":        self._rule_map,
                "seq":        self._rule_seq,
            }
            rules.update(self._terminals)
            self.scanner.consume('(')
            key = self.scanner.onToken(self._terminals, consume=True)
            self.scanner.consume(',')
            value = self.scanner.onToken(rules, consume=True)
            self.scanner.consume(')')
            return {"map": (key, value)}

        def _rule_seq(self):
            rules = {
                "tensor":     self._rule_tensor,
                "map":        self._rule_map,
                "seq":        self._rule_seq
            }
            rules.update(self._terminals)
            self.scanner.consume('(')
            result = self.scanner.onToken(rules, consume=True)
            self.scanner.consume(')')
            return {"seq": result}

        def parse(self):
            rules = {
                "tensor":     self._rule_tensor,
                "map":        self._rule_map,
                "seq":        self._rule_seq,
            }
            return self.scanner.onToken(rules, consume=True)

    def __init__(self, typeStr):
        super()
        self.original = typeStr
        scanner = self._Scanner(typeStr)
        parser = self._Parser(scanner)
        self.update(parser.parse())

    def __str__(self):
        return self._text_walkParseTree(self)

    def __repr__(self):
        return f"OnnxType({self.original.__repr__()})"

    def _text_walkParseTree(self, node):
        if isinstance(node,str):
            return node.replace("_","")
        elif isinstance(node,dict):
            subresults = []
            for key,val in node.items():
                subresults.append(key + "_" + self._text_walkParseTree(val))
            return "__".join(subresults)
        elif isinstance(node,tuple):
            return "__".join([ self._text_walkParseTree(t) for t in node ])
        else:
            raise BaseException(f"unknown parseTree item: '{node}'")

    def onnxTensorDataTypes(self):
        results = []
        self._onnxTensorDataType_walkParseTree(self, results)
        return list(filter(None,results))

    def _onnxTensorDataType_walkParseTree(self, node, results):
        if isinstance(node,str):
            results.append(None)
        elif isinstance(node,dict):
            for key,val in node.items():
                if key == "tensor":
                    results.append(self._onnxTensorDataType[val])
                else:
                    self._onnxTensorDataType_walkParseTree(val,results)
        elif isinstance(node,tuple):
            for val in node:
                self._onnxTensorDataType_walkParseTree(val, results)
        else:
            raise BaseException(f"unknown parseTree item: '{node}'")

    def __hash__(self):
        return self.original.__hash__()

class OnnxTypeList(list):
    def __init__(self, typeList):
        super()
        types = []
        types.extend(typeList)
        types.sort()
        self.extend([OnnxType(t) for t in types])

    def __str__(self):
        return ", ".join([f"{t}" for t in self])

    def __repr__(self):
        types = ", ".join([t.original.__repr__() for t in self])
        return f"OnnxTypeList([{types}])"

class OnnxConstraint():
    def __init__(self, constraint, input=False, output=False):
        if isinstance(constraint, dict):
            self.types = constraint['types']
            self.description = constraint['description']
            self.name = constraint['name']
            self.input = constraint['input']
            self.output = constraint['output']
        else:
            self.types = OnnxTypeList(constraint.allowed_type_strs)
            self.description = constraint.description
            self.name = constraint.type_param_str
            self.input = input
            self.output = output

    def text(self, prefix=""):
        lines = []
        lines.append(f"{prefix}Constraint {self.name}:")
        lines.append(format_text(prefix + "  ", "", [self.description]))
        lines.append(format_text(prefix + "  ", "Allowed Types:", [str(self.types)] ))
        return "\n".join(lines)

    def __repr__(self):
        return f"OnnxConstraint({self.__dict__.__repr__()})"

class OnnxConstraints(dict):
    def __init__(self, schema):
        super()
        constraints = {c.type_param_str for c in schema.type_constraints}
        inputs = {i.typeStr for i in schema.inputs if i.typeStr in constraints}
        outputs = {o.typeStr for o in schema.outputs if o.typeStr in constraints}
        for constraint in schema.type_constraints:
            self[constraint.type_param_str] = OnnxConstraint(constraint, input=constraint.type_param_str in inputs, output=constraint.type_param_str in outputs)

    def typePermutations(self, filterInput=False, filterOutput=False):
        return list(filter(None,(self.typePermutationText(p) for p in self.typePermutationsTuple(filterInput,filterOutput))))

    def typePermutationText(self, permutation):
        return "__".join([ f"{x[0]}_{x[1]}" for x in permutation ])

    def typePermutationsTuple(self, filterInput=False, filterOutput=False):
        # a implies b is the same as bool(a) ** bool(b)
        values = filter(lambda x: (x.input ** filterInput) and (x.output ** filterOutput), self.values())
        tuples = [list(map(lambda x: (c.name,x), c.types)) for c in values]
        return itertools.product(*tuples)

    def typePermutationsMap(self, filterInput=False, filterOutput=False):
        result = {}
        for permutation in self.typePermutationsTuple(filterInput, filterOutput):
            tmp = result
            constraints = []
            for constraint in permutation:
                constraints.append(constraint)
                tmp = tmp.setdefault(tuple(constraints), {})
        return result

    def text(self, prefix=""):
        paragraphs = [ c.text(prefix) for c in self.values() ]
        return f"\n{prefix}\n".join(paragraphs)

    def __str__(self):
        return self.text()

class OnnxAttribute():
    _onnxAttributeDataType = {
        "UNDEFINED"      : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED",
        "FLOAT"          : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT",
        "INT"            : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT",
        "STRING"         : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING",
        "TENSOR"         : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR",
        "GRAPH"          : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH",
        "SPARSE_TENSOR"  : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR",
        "FLOATS"         : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS",
        "INTS"           : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS",
        "STRINGS"        : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS",
        "TENSORS"        : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS",
        "GRAPHS"         : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS",
        "SPARSE_TENSORS" : "ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS",
    }
    _onnxAttributeDataTypeCDecl = {
        "FLOAT"          : [("f","float","{name}")],
        "INT"            : [("i","int64_t","{name}")],
        "STRING"         : [("s","char*","{name}")],
        "TENSOR"         : [("t","Onnx__TensorProto*","{name}")],
        "GRAPH"          : [("g","Onnx__GraphProto*","{name}")],
        "SPARSE_TENSOR"  : [("sparse_tensor","Onnx__SparseTensorProto*","{name}")],
        "FLOATS"         : [("n_floats","size_t","n_{name}"),("floats","float*","{name}")],
        "INTS"           : [("n_ints","size_t","n_{name}"),("ints","int64_t*","{name}")],
        "STRINGS"        : [("n_strings","size_t","n_{name}"),("strings","char**","{name}")],
        "TENSORS"        : [("n_tensors","size_t","n_{name}"),("tensors","Onnx__TensorProto**","{name}")],
        "GRAPHS"         : [("n_graphs","size_t","n_{name}"),("graphs","Onnx__GraphProto**","{name}")],
        "SPARSE_TENSORS" : [("n_sparse_tensors","size_t","n_{name}"),("sparse_tensors","Onnx__SparseTensorProto**","{name}")],
    }

    def __init__(self, name, attribute):
        self.name = name
        if isinstance(attribute, dict):
            self.optional = attribute['optional']
            self.type = attribute['type']
            self.description = attribute['description']
        else:
            self.optional = not attribute.required
            self.type = attribute.type.name
            self.description = attribute.description

    def text(self, prefix=""):
        lines = []
        lines.append(f"{prefix}Attribute {self.type} {self.name} {'(optional)'*self.optional}:")
        lines.append(format_text(prefix + "  ", None, [self.description]))
        return "\n".join(lines)

    def onnxAttributeDataType(self):
        return self._onnxAttributeDataType[self.type]

    def onnxAttributeDataTypeCDecl(self):
        result = []
        for decls in self._onnxAttributeDataTypeCDecl[self.type]:
            result.append((s.format(name = self.name) for s in decls ))
        return result

    def __repr__(self):
        attribute = self.__dict__.copy()
        del attribute['name']
        return f"OnnxAttribute({self.name.__repr__()}, {attribute.__repr__()})"

    def __str__(self):
        return self.text()


class OnnxAttributeList(list):
    def __init__(self, schema):
        super()
        for name,attribute in schema.attributes.items():
            self.append(OnnxAttribute(name, attribute))

    def text(self, prefix=""):
        paragraphs = [ a.text(prefix) for a in self ]
        return f"\n{prefix}\n".join(paragraphs)

    def __str__(self):
        return self.text()

class OnnxInput():
    def __init__(self, input):
        if isinstance(input, dict):
            self.name = input['name']
            self.description = input['description']
            self.isHomogeneous = input['isHomogeneous']
            self.optional = input['optional']
            self.variadic = input['variadic']
            self.constraint = input['constraint']
            self.types = input['types']
        else:
            self.name = input.name
            self.description = input.description.strip()
            self.isHomogeneous = input.isHomogeneous
            self.optional = (input.option.name == "Optional")
            self.variadic = (input.option.name == "Variadic")
            self.constraint = input.typeStr
            self.types = OnnxTypeList(input.types)

    def text(self, prefix=""):
        lines = []
        lines.append(f"{prefix}Input {self.constraint} {self.name}:")
        lines.append(format_text(prefix + "  ", "", [self.description]))
        lines.append(format_text(prefix + "  ", "Allowed Types:", [str(self.types)] ))
        return "\n".join(lines)

    def __repr__(self):
        return f"OnnxInput({self.__dict__.__repr__()})"

    def __str__(self):
        return self.text()


class OnnxInputList(list):
    def __init__(self, schema):
        super()
        self.extend([ OnnxInput(i) for i in schema.inputs])

    def text(self, prefix=""):
        paragraphs = [ i.text(prefix) for i in self ]
        return f"\n{prefix}\n".join(paragraphs)

    def __str__(self):
        return self.text()


class OnnxOutput():
    def __init__(self, output):
        if isinstance(output, dict):
            self.name = output['name']
            self.description = output['description']
            self.isHomogeneous = output['isHomogeneous']
            self.optional = output['optional']
            self.variadic = output['variadic']
            self.constraint = output['constraint']
            self.types = output['types']
        else:
            self.name = output.name
            self.description = output.description
            self.isHomogeneous = output.isHomogeneous
            self.optional = (output.option.name == "Optional")
            self.variadic = (output.option.name == "Variadic")
            self.constraint = output.typeStr
            self.types = OnnxTypeList(output.types)

    def text(self, prefix=""):
        lines = []
        lines.append(f"{prefix}Output {self.constraint} {self.name}:")
        lines.append(format_text(prefix + "  ", None, [self.description]))
        lines.append(format_text(prefix + "  ", "Allowed Types:", [str(self.types)] ))
        return "\n".join(lines)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.text()

class OnnxOutputList(list):
    def __init__(self, schema):
        super()
        self.extend([ OnnxOutput(i) for i in schema.outputs])

    def text(self, prefix=""):
        paragraphs = [ i.text(prefix) for i in self ]
        return f"\n{prefix}\n".join(paragraphs)

    def __str__(self):
        return self.text()

class OnnxDoc():
    def __init__(self, schema):
        if isinstance(schema,str):
            self.doc = schema
        else:
            self.doc = schema.doc.strip()

    def __repr__(self):
        return f"OnnxDoc({self.doc.__repr__()})"

    def text(self, prefix=" * "):
        return prefix + f"\n{prefix}".join(self.doc.split('\n'))

    def __str__(self):
        return self.text()

class OnnxSchema():

    def __init__(self, schema):
        self.name = None
        self.doc = None
        self.deprecated = None
        self.operator_name = None
        self.version = None
        self.domain = None
        self.constraints = None
        self.attributes = None
        self.inputs = None
        self.outputs = None
        self.ref_doc = None
        self.range_input = None
        self.range_output = None
        self.ref_file = None
        self._schema = None
        if isinstance(schema, dict):
            self.__dict__.update(schema)
        else:
            self.doc = OnnxDoc(schema)
            self.name = schema.name
            self.deprecated = schema.deprecated
            self.operator_name = self._operator_name(schema)
            self.version = schema.since_version
            self.domain = self._domain(schema)
            self.constraints = OnnxConstraints(schema)
            self.attributes = OnnxAttributeList(schema)
            self.inputs = OnnxInputList(schema)
            self.outputs = OnnxOutputList(schema)
            self.ref_doc = self._ref_doc(schema)
            self.range_input = (schema.min_input, schema.max_input)
            self.range_output = (schema.min_output, schema.max_output)
            self.ref_file = (schema.file,schema.line)
            self._schema = schema

    def __repr__(self):
      return f"OnnxSchema({self.__dict__.__repr__()})"

    def _operator_name(self, schema, name=True, version=True):
        opname = f"operator__{self._domain(schema)}"
        if name:
            opname += f"__{schema.name}"
        if version:
            opname += f"__{schema.since_version}"
        return re.sub(r"\W", "_", opname).lower()

    def _domain(self, schema):
        domain = "ai.onnx"
        if schema.domain:
            domain = schema.domain
        return domain.strip()

    def _ref_doc(self, schema):
        domain = self._domain(schema)
        if domain == 'ai.onnx':
            return f"https://github.com/onnx/onnx/blob/master/docs/Operators.md#{schema.name}"
        elif domain == 'ai.onnx.ml':
            return f"https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#{schema.name}"
        else:
            return ''
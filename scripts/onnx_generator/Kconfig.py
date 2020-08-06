from .Template import Template
import re

class Domain(Template):
    _basepath = "{path}"
    _filepath = "{domain}/Kconfig"
    _template = '''
menuconfig HAVE_DOMAIN__{domain_sane_upper}
	bool "{domain}"
	help
		Enable the operator domain '{domain}'
	default {enabled}
	if HAVE_DOMAIN__{domain_sane_upper}
		rsource "*/Kconfig"
	endif
'''
    def __init__(self, domain, path, enabled=True):
        self.domain = domain
        self.path = path
        self.enabled = "y" if (enabled != "n" and enabled) else "n"
        self.domain_sane_upper = re.sub(r"\W", "_", domain).upper()


class Operator(Template):
    _basepath = "{path}"
    _filepath = "{domain}/{operator}/Kconfig"
    _template = '''
menuconfig HAVE_OPERATOR__{domain_sane_upper}__{operator_sane_upper}
	bool "{operator}"
	default {enabled}
	help
		Enable the domain '{domain}' operator '{operator}'
	if HAVE_OPERATOR__{domain_sane_upper}__{operator_sane_upper}
		rsource "*/Kconfig"
	endif
'''
    def __init__(self, domain, operator, path, enabled=True):
        self.domain = domain
        self.operator = operator
        self.path = path
        self.enabled = "y" if (enabled != "n" and enabled) else "n"
        self.domain_sane_upper = re.sub(r"\W", "_", domain).upper()
        self.operator_sane_upper = re.sub(r"\W", "_", operator).upper()


class Operator_version(Template):
    _basepath = "{path}"
    _filepath = "{domain}/{operator}/{version}/Kconfig"
    _template = '''
menuconfig HAVE_OPERATOR__{domain_sane_upper}__{operator_sane_upper}__{version}
	bool "{version}"
	help
		Enable domain '{domain}' operator '{operator}' version '{version}'
	default {enabled}
	if HAVE_OPERATOR__{domain_sane_upper}__{operator_sane_upper}__{version}
		rsource "*.Kconfig"
	endif
'''
    def __init__(self, domain, operator, version, path, enabled=True):
        self.domain = domain
        self.operator = operator
        self.path = path
        self.enabled = "y" if (enabled != "n" and enabled) else "n"
        self.domain_sane_upper = re.sub(r"\W", "_", domain).upper()
        self.operator_sane_upper = re.sub(r"\W", "_", operator).upper()
        self.version = version

class Operator_version_type(Template):
    _template = '''
config HAVE_TYPE_{kind_upper}__{domain_sane_upper}__{operator_sane_upper}__{version}__{type_upper}
    bool "{type}"
    depends on HAVE_TYPE_{kind_upper}__{type_upper}
	help
		Enable {kind_lower} '{type}' type for domain '{domain}' operator '{operator}' version '{version}'.
	default {enabled}
'''
    def __init__(self, domain, operator, version, kind, type, enabled=True):
        self.domain = domain
        self.operator = operator
        self.version = version
        self.kind = kind
        self.type = type
        self.enabled = "y" if (enabled != "n" and enabled) else "n"
        self.domain_sane_upper = re.sub(r"\W", "_", domain).upper()
        self.operator_sane_upper = re.sub(r"\W", "_", operator).upper()
        self.kind_upper = kind.upper()
        self.kind_lower = kind.lower()
        self.type_upper = type.upper()

class Operator_version_types(Template):
    _basepath = "{path}"
    _filepath = "{domain}/{operator}/{version}/{kind}.Kconfig"
    _template = '''
menu "{kind} types"
{options}
endmenu
'''
    def __init__(self, domain, operator, version, kind, types, path):
        self.domain = domain
        self.operator = operator
        self.version = version
        self.path = path
        self.kind = kind
        self.types = types
        self.options = "\n".join([ str(Operator_version_type(domain, operator, version, kind, t)) for t in types])

class Global(Template):
    _basepath = "{path}"
    _filepath = "Kconfig"
    _template = '''
rsource "*/Kconfig"
rsource "*.Kconfig"
'''
    def __init__(self, path):
        self.path = path

class Global_type(Template):
    _template = '''
config HAVE_TYPE_{kind_upper}__{type_upper}
    bool "{type}"
	help
		Enable {kind_lower} '{type}' globally.
	default {enabled}
'''
    def __init__(self, kind, type, enabled=True):
        self.kind = kind
        self.type = type
        self.enabled = "y" if (enabled != "n" and enabled) else "n"
        self.kind_upper = kind.upper()
        self.kind_lower = kind.lower()
        self.type_upper = type.upper()

class Global_types(Template):
    _basepath = "{path}"
    _filepath = "{kind}.Kconfig"
    _template = '''
menu "global {kind} types"
{options}
endmenu
'''
    def __init__(self, kind, types, path):
        self.path = path
        self.kind = kind
        self.types = types
        self.options = "\n".join([ str(Global_type(kind, t)) for t in types])

class Kconfigs:
    def __init__(self, schemas, path):
        self.schemas = schemas
        self.path = path

    def __iter__(self):
        yield Global(self.path)

        all_input_types = set((str(t) for s in self.schemas for i in s.inputs for t in i.types))
        all_output_types = set((str(t) for s in self.schemas for o in s.outputs for t in o.types))
        all_constraint_types = set((str(t) for s in self.schemas for c in s.constraints.values() for t in c.types))
        all_tensor_types = set.union(all_input_types,all_output_types,all_constraint_types)
        yield Global_types("tensor", all_tensor_types, self.path)

        all_attribute_types = set((str(a.type) for s in self.schemas for a in s.attributes))
        yield Global_types("attribute", all_attribute_types, self.path)

        domains = set((s.domain for s in self.schemas))
        for d in domains:
            yield Domain(d,self.path)

        domain_operators = set(((s.domain,s.name) for s in self.schemas))
        for d,op in domain_operators:
            yield Operator(d,op,self.path)

        for s in self.schemas:
            d = s.domain
            op = s.name
            v = s.version
            yield Operator_version(d,op,v, self.path)
            attribute_types = set((str(a.type) for a in s.attributes))
            if attribute_types:
                yield Operator_version_types(d,op,v,"attribute",attribute_types, self.path)
            input_types = set((str(t) for i in s.inputs for t in i.types))
            output_types = set((str(t) for o in s.outputs for t in o.types))
            constraint_types = set((str(t) for c in s.constraints.values() for t in c.types))
            tensor_types = set.union(input_types,output_types,constraint_types)
            if tensor_types:
                yield Operator_version_types(d,op,v,"tensor",tensor_types, self.path)

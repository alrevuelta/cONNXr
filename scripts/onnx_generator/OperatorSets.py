import inspect
import os
import pathlib
import re

class OperatorSetEntry:
    _template = '''
{{
  .name = "{name}",
  .resolver = (operator_resolver) &resolve_{operator_name}
}}
'''
    def __init__(self, schema):
        self.schema = schema

    def text(self):
        return self._template.format(
            operator_name = self.schema.operator_name,
            name = self.schema.name
        ).strip()

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schema.__repr__()})"


class OperatorSet:
    _template = '''
operator_set {name} = {{
  .version = {version},
  .domain  = "{domain}",
  .length  = {length},
  .entries = {{
    {entries}
  }}
}};
'''

    def __init__(self, domain, version, schemas):
        self.domain = domain
        self.domain_sane = re.sub("\W","_",domain)
        self.version = version
        self.schemas = schemas
        self.name = f"operator_set__{self.domain_sane}__{version}"

    def text(self):
        return self._template.format(
            entries = ",".join([ str(OperatorSetEntry(s)) for s in self.schemas ]),
            version =self.version,
            domain  = self.domain,
            domain_sane = self.domain_sane,
            length = len(self.schemas),
            name = self.name
        ).strip()
    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.domain.__repr__()}, {self.version.__repr__()}, {self.schemas.__repr__()})"


class OperatorSets:
    _template = '''
//this file was generated by {script}
#include "operators/operator_sets.h"

{includes}

{sets}

operator_sets all_operator_sets = {{
  .length = {length},
  .sets   = {{
    {set_refs}
  }}
}};
'''

    def __init__(self, headers, path):
        self.headers = headers
        self.path = path

    def text(self):
        sets = []
        versions = set()
        domain2name2version2schema = {}
        for header in self.headers:
            schema = header.schema
            name2version2schema = domain2name2version2schema.setdefault(schema.domain,{})
            name2version2schema.setdefault(schema.name,{})[schema.version] = schema
            versions.add(schema.version)

        for version in versions:
            for domain, name2version2schema in domain2name2version2schema.items():
                tmp = []
                for name, version2schema in name2version2schema.items():
                    for v in range(version, 0, -1):
                        if v in version2schema:
                            tmp.append(version2schema[v])
                            break;
                sets.append(OperatorSet(domain, version, tmp))


        return self._template.format(
            includes = "\n".join([ f'#include "{h.filename("operators")}"' for h in self.headers ]),
            script=self._rel_path(inspect.getfile(inspect.currentframe())),
            sets = "\n\n".join([ str(s) for s in sets]),
            length = len([ str(s) for s in sets]),
            set_refs = ",\n".join([f"&{s.name}" for s in sets])
        )

    def filename(self):
        path = str(self.path)
        path += f"/operator_sets.c"
        return pathlib.Path(path)

    def _rel_path(self, path):
        return os.path.relpath(os.path.realpath(path),os.path.realpath(self.path))

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"{self.__class__}({self.schemas.__repr__()}, {self.path.__repr__()})"

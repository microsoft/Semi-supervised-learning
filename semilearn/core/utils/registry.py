# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib

__all__ = [
    'ALGORITHMS',
    'IMB_ALGORITHMS',
]

class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            print("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


ALGORITHMS = Register('algorithms')
IMB_ALGORITHMS = Register('imb_algorithms')



def _handle_errors(errors):
    """
    Log out and possibly reraise errors during import.
    """
    if not errors:
        return
    
    for name, err in errors:
        print("Module {} import failed: {}".format(name, err))


ALL_MODULES = [
    # NOTE: add all algorithms here
    ('semilearn.algorithms', ['adamatch', 'comatch', 'crmatch', 'dash', 'fixmatch', 'flexmatch', 'fullysupervised', 'meanteacher',
                              'mixmatch', 'pimodel', 'pseudolabel', 'remixmatch', 'simmatch', 'uda', 'vat', 'softmatch', 'freematch', 'defixmatch']),
    ('semilearn.imb_algorithms', ['abc', 'cossl', 'adsh', 'crest', 'darp', 'daso', 'debiaspl', 'saw', 'tras'])
]


def import_all_modules_for_register():
    """
    Import all modules for register.
    """
    all_modules = ALL_MODULES
    errors = []
    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)


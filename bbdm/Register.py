import logging
#import importlib
#import os


class Register:
    def __init__(self, registry_name):
        self.dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable")
        if key is None:
            key = value.__name__
        if key in self.dict:
            logging.warning("Key %s already in registry %s." % (key, self.__name__))
        self.dict[key] = value

    def register_with_name(self, name):
        def register(target):
            if callable(target):
                self[name] = target
                return target
            else:
                raise TypeError(f"Error: Target '{target}' is not callable!")

        return register

    def __getitem__(self, key):
        if key not in self.dict:
            raise KeyError(f"KeyError: '{key}' not found in Register '{self._name}'")
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()


class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    datasets = Register('datasets')
    runners = Register('runners')

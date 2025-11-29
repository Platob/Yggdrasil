from abc import ABC

from ..abstract_field import AbstractField, PythonField, SparkField, ArrowField


class AbstractScalarField(AbstractField, ABC):
    pass


class PythonScalarField(PythonField, ABC):
    pass
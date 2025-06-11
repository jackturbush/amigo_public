from .amigo import Vector, OptimizationProblem
from .component import Component
from .unary_operations import *
from .model import Model


def get_include():
    from .amigo import AMIGO_INCLUDE_PATH

    return AMIGO_INCLUDE_PATH


def get_a2d_include():
    from .amigo import A2D_INCLUDE_PATH

    return A2D_INCLUDE_PATH

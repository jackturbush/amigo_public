import numpy as np
import ast
import importlib
from .amigo import VectorInt, OptimizationProblem
from .component import Component
from typing import Self
from collections import defaultdict


def _import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _parse_var_expr(expr):
    """
    Parse an expression like:
      - 'model.group.var'
      - 'model.group.var[:]'
      - 'model.group.var[:, 0]'

    path = ["model", "group", "var"]
    indices = None or tuple of slice/index

    Returns:
        path, indices
    """
    try:
        node = ast.parse(expr, mode="eval").body

        # Determine if it's an attribute or a subscript
        if isinstance(node, ast.Subscript):
            attr_node = node.value
            indices = _parse_indices(node.slice)
        elif isinstance(node, ast.Attribute):
            attr_node = node
            indices = None
        else:
            raise ValueError("Unsupported expression format.")

        # Extract the full attribute path
        path = _parse_attribute_path(attr_node)

        return path, indices
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr}': {e}")


def _parse_attribute_path(attr):
    path = []
    while isinstance(attr, ast.Attribute):
        path.append(attr.attr)
        attr = attr.value
    if isinstance(attr, ast.Name):
        path.append(attr.id)
    else:
        raise ValueError("Invalid attribute chain")
    return list(reversed(path))


def _parse_indices(slice_node):
    if isinstance(slice_node, ast.Tuple):
        return tuple(_eval_ast_index(elt) for elt in slice_node.elts)
    else:
        return (_eval_ast_index(slice_node),)


def _eval_ast_index(node):
    if isinstance(node, ast.Slice):
        return slice(
            _eval_ast_index(node.lower) if node.lower else None,
            _eval_ast_index(node.upper) if node.upper else None,
            _eval_ast_index(node.step) if node.step else None,
        )
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name) and node.id == "None":
        return None
    else:
        return ast.literal_eval(node)


class AliasTracker:
    def __init__(self):
        self.parent = {}

    def _find(self, var):
        # Initialize if not seen
        if var not in self.parent:
            self.parent[var] = var
        # Path compression
        if self.parent[var] != var:
            self.parent[var] = self._find(self.parent[var])
        return self.parent[var]

    def alias(self, var1, var2):
        root1 = self._find(var1)
        root2 = self._find(var2)
        if root1 != root2:
            self.parent[root2] = root1  # or vice versa

    def get_alias_group(self, var):
        root = self._find(var)
        return [v for v in self.parent if self._find(v) == root]

    def all_groups(self):
        groups = defaultdict(list)
        for var in self.parent:
            root = self._find(var)
            groups[root].append(var)
        return list(groups.values())


class ComponentSet:
    def __init__(self, name: str, size: int, comp_obj, var_shapes):
        self.name = name
        self.size = size
        self.comp_obj = comp_obj
        self.class_name = comp_obj.name
        self.vars = {}
        for var_name, shape in var_shapes.items():
            self.vars[var_name] = -np.ones(shape, dtype=int)

    def get_var(self, varname):
        return self.vars[varname]

    def create_model(self, module_name: str):
        size = 0
        dim = 0
        for name in self.vars:
            shape = self.vars[name].shape
            size += np.prod(shape)

            if len(shape) == 1:
                dim += 1
            else:
                dim += np.prod(shape[1:])

        # Set the entries of the vectors
        vec = VectorInt(size)
        array = vec.get_array()

        offset = 0
        for name in self.vars:
            shape = self.vars[name].shape
            if len(shape) == 1:
                array[offset::dim] = self.vars[name][:]
                offset += 1
            elif len(shape) == 2:
                for i in range(shape[1]):
                    array[offset::dim] = self.vars[name][:, i]
                    offset += 1
            elif len(shape) == 3:
                for i in range(shape[1]):
                    for i in range(shape[2]):
                        array[offset::dim] = self.vars[name][:, i, j]
                        offset += 1

        # Create the object
        return _import_class(module_name, self.class_name)(vec)


class Model:
    def __init__(self, module_name):
        self.module_name = module_name
        self.comp = {}
        # self.index_pool = GlobalIndexPool()
        # self.tracker = SliceAliasTracker()
        self.connections = []

    def generate_cpp(self):
        """
        Generate the C++ header and pybind11 wrapper for the model.
        """

        # C++ file contents
        cpp = '#include "a2dcore.h"\n'
        cpp += "namespace amigo {"

        # pybind11 file contents
        py11 = "#include <pybind11/numpy.h>\n"
        py11 += "#include <pybind11/pybind11.h>\n"
        py11 += "#include <pybind11/stl.h>\n"
        py11 += '#include "serial_component_set.h"\n'
        py11 += f'#include "{self.module_name}.h"\n'
        py11 += "namespace py = pybind11;\n"

        mod_ident = "mod"
        py11 += f"PYBIND11_MODULE({self.module_name}, {mod_ident}) " + "{\n"

        # Write out the classes needed - class names must be unique
        # so we don't duplicate code
        class_names = {}
        for name in self.comp:
            class_name = self.comp[name].class_name
            if class_name not in class_names:
                class_names[class_name] = True

                # Generate the C++
                cpp += self.comp[name].comp_obj.generate_cpp()

                py11 += (
                    self.comp[name].comp_obj.generate_pybind11(mod_ident=mod_ident)
                    + ";\n"
                )

        cpp += "}\n"
        py11 += "}\n"

        filename = self.module_name + ".h"
        with open(filename, "w") as fp:
            fp.write(cpp)

        filename = self.module_name + ".cpp"
        with open(filename, "w") as fp:
            fp.write(py11)

        return

    def add_component(self, name: str, size: int, comp_obj: Component):
        """
        Add a single component under the
        """
        if name in self.comp:
            raise ValueError(f"Cannot add two components with the same name")

        var_shapes = comp_obj.get_var_shapes()

        for var_name in var_shapes:
            if var_shapes[var_name] is None:
                var_shapes[var_name] = (size,)
            elif isinstance(var_shapes[var_name], tuple):
                var_shapes[var_name] = (size,) + var_shapes[var_name]
            else:
                var_shapes[var_name] = (size, var_shapes[var_name])

        self.comp[name] = ComponentSet(name, size, comp_obj, var_shapes)

        return

    def add_model(self, name: str, model: Self):
        """
        Add the given model as a sub-model
        """
        if name in self.comp:
            raise ValueError(
                f"Cannot add a sub-model with the same name as a component"
            )

        # Add all of the sub-model components
        for comp_name in model.comp:
            sub_name = name + "." + comp_name
            sub_size = model.comp[comp_name].size
            sub_obj = model.comp[comp_name].comp_obj

            self.add_component(sub_name, sub_size, sub_obj)

        # Add all of the sub-model connections (if any exist)
        for src_expr, tgt_expr in model.connections:
            self.connect(name + "." + src_expr, name + "." + tgt_expr)

        return

    def connect(self, src_expr: str, tgt_expr: str):
        # Add the connection for later use
        self.connections.append((src_expr, tgt_expr))
        return

    def _normalize_slice_spec(self, slice_spec, shape):
        """
        Converts a slice spec (int, slice, or tuple of those) into a list of ranges,
        compatible with the given shape (1D or 2D).
        """
        ndim = len(shape)

        # Ensure it's a tuple
        if slice_spec is None:
            slice_spec = (slice(None),) * ndim
        elif not isinstance(slice_spec, tuple):
            slice_spec = (slice_spec,)

        # Pad with slice(None) if necessary
        slice_spec += (slice(None),) * (ndim - len(slice_spec))

        if len(slice_spec) != ndim:
            raise ValueError(
                f"Slice spec dimension {len(slice_spec)} does not match shape {ndim}"
            )

        # Build ranges
        ranges = []
        for s, dim in zip(slice_spec, shape):
            if isinstance(s, int):
                s = s if s >= 0 else dim + s
                if s < 0 or s >= dim:
                    raise IndexError(
                        f"Index {s} out of bounds for axis with size {dim}"
                    )
                ranges.append(range(s, s + 1))
            elif isinstance(s, slice):
                ranges.append(range(*s.indices(dim)))
            else:
                raise TypeError(f"Unsupported index type: {s}")

        return ranges

    def _get_var_connections(self, a_var, a_slice, a_shape, b_var, b_slice, b_shape):
        a_ranges = self._normalize_slice_spec(a_slice, a_shape)
        b_ranges = self._normalize_slice_spec(b_slice, b_shape)

        a_indices = list(np.ndindex(*[len(r) for r in a_ranges]))
        b_indices = list(np.ndindex(*[len(r) for r in b_ranges]))

        if len(a_indices) != len(b_indices):
            raise ValueError("Sliced regions are not the same size")

        def format_index(name, ranges, idx):
            actual_indices = [r[i] for r, i in zip(ranges, idx)]
            return f"{name}[{', '.join(map(str, actual_indices))}]"

        result = []
        for a_idx, b_idx in zip(a_indices, b_indices):
            a_str = format_index(a_var, a_ranges, a_idx)
            b_str = format_index(b_var, b_ranges, b_idx)
            result.append((a_str, b_str))

        return result

    def initialize(self):
        """
        Initialize the variable indices for each component
        """

        tracker = AliasTracker()
        for a_expr, b_expr in self.connections:
            a_path, a_slice = _parse_var_expr(a_expr)
            a_var = ".".join(a_path)
            a_shape = self.get_vars(a_var).shape

            b_path, b_slice = _parse_var_expr(b_expr)
            b_var = ".".join(b_path)
            b_shape = self.get_vars(b_var).shape

            for a, b in self._get_var_connections(
                a_var, a_slice, a_shape, b_var, b_slice, b_shape
            ):
                tracker.alias(a, b)

        # Order the variables
        counter = 0

        # Order any aliased variables
        groups = tracker.all_groups()
        for group in groups:
            for expr in group:
                path, slice = _parse_var_expr(expr)
                self.get_vars(".".join(path))[slice] = counter

            counter += 1

        for name, comp in self.comp.items():
            for varname, array in comp.vars.items():
                arr = array.ravel()

                for i in range(arr.shape[0]):
                    if arr[i] == -1:
                        arr[i] = counter
                        counter += 1

        self.num_variables = counter

        return

    def get_vars(self, name: str):
        path, indices = _parse_var_expr(name)
        comp_name = ".".join(path[:-1])
        var_name = path[-1]

        if indices is None:
            return self.comp[comp_name].get_var(var_name)
        else:
            return self.comp[comp_name].get_var(var_name)[indices]

    def create_opt_problem(self):
        objs = []
        for name, comp in self.comp.items():
            objs.append(comp.create_model(self.module_name))

        return OptimizationProblem(objs)

    def print_indices(self):
        for comp_name, comp in self.comp.items():
            print(f"Component: {comp_name}")
            for varname, array in comp.vars.items():
                print(f"  {varname}:\n{array}")

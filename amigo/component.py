import math


class ExprNode:
    def __init__(self):
        self.name = None  # For output variable names

    def evaluate(self, env):
        raise NotImplementedError

    def generate_cpp(self):
        raise NotImplementedError


class ConstNode(ExprNode):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, env):
        return self.value

    def generate_cpp(self, index=None):
        return str(self.value)


class VarNode(ExprNode):
    def __init__(self, name, shape=None, type=float):
        super().__init__()
        self.name = name
        self.shape = shape  # e.g. (M, N) for 2D
        self.type = type

    def generate_cpp(self, index=None):
        if index is None:
            return self.name
        elif isinstance(index, tuple):
            # For 2D index like A[i][j]
            i_str = index[0].generate_cpp()
            j_str = index[1].generate_cpp()
            return f"{self.name}[{i_str}][{j_str}]"
        else:
            return f"{self.name}[{index.generate_cpp()}]"


class IndexNode(ExprNode):
    def __init__(self, array_node, index_node):
        super().__init__()
        self.array_node = array_node
        self.index_node = index_node  # can be ExprNode or tuple of ExprNodes

    def evaluate(self, env):
        array_val = self.array_node.evaluate(env)
        if isinstance(self.index_node, tuple):
            idx = tuple(i.evaluate(env) for i in self.index_node)
            return array_val[idx[0]][idx[1]]
        else:
            idx = self.index_node.evaluate(env)
            return array_val[idx]

    def generate_cpp(self, index=None):
        arr = self.array_node.generate_cpp()
        if isinstance(self.index_node, tuple):
            idxs = tuple(i.generate_cpp(index) for i in self.index_node)
            return f"{arr}[{idxs[0]}][{idxs[1]}]"
        else:
            idx = self.index_node.generate_cpp(index)
            return f"{arr}[{idx}]"


class OpNode(ExprNode):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, env):
        lval = self.left.evaluate(env)
        rval = self.right.evaluate(env)
        # Broadcast-aware evaluation
        if hasattr(lval, "__len__") and hasattr(rval, "__len__"):
            return [self._op(a, b) for a, b in zip(lval, rval)]
        elif hasattr(lval, "__len__"):
            return [self._op(a, rval) for a in lval]
        elif hasattr(rval, "__len__"):
            return [self._op(lval, b) for b in rval]
        else:
            return self._op(lval, rval)

    def _op(self, a, b):
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[self.op]

    def generate_cpp(self, index=None):
        a = self.left.generate_cpp(index)
        b = self.right.generate_cpp(index)
        return f"({a} {self.op} {b})"


class UnaryNode(ExprNode):
    def __init__(self, func_name, func, operand):
        super().__init__()
        self.func_name = func_name
        self.func = func
        self.operand = operand

    def evaluate(self, env):
        val = self.operand.evaluate(env)
        if hasattr(val, "__len__"):
            return [self.func(v) for v in val]
        return self.func(val)

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"{self.func_name}({a})"


class UnaryNegNode(ExprNode):
    def __init__(self, operand):
        super().__init__()
        self.operand = operand

    def evaluate(self, env):
        val = self.operand.evaluate(env)
        if hasattr(val, "__len__"):
            return [-v for v in val]
        return -val

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"-({a})"


class Expr:
    def __init__(self, node: ExprNode):
        self.node = node

    def __neg__(self):
        return Expr(UnaryNegNode(self.node))

    def __add__(self, other):
        return Expr(OpNode("+", self.node, self._to_node(other)))

    def __sub__(self, other):
        return Expr(OpNode("-", self.node, self._to_node(other)))

    def __mul__(self, other):
        return Expr(OpNode("*", self.node, self._to_node(other)))

    def __truediv__(self, other):
        return Expr(OpNode("/", self.node, self._to_node(other)))

    def __radd__(self, other):
        return Expr(OpNode("+", self._to_node(other), self.node))

    def __rsub__(self, other):
        return Expr(OpNode("-", self._to_node(other), self.node))

    def __rmul__(self, other):
        return Expr(OpNode("*", self._to_node(other), self.node))

    def __rtruediv__(self, other):
        return Expr(OpNode("/", self._to_node(other), self.node))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx_node = tuple(self._to_node(i) for i in idx)
        else:
            idx_node = self._to_node(idx)
        return Expr(IndexNode(self.node, idx_node))

    def evaluate(self, env):
        return self.node.evaluate(env)

    def generate_cpp(self):
        return self.node.generate_cpp()

    def _to_node(self, val):
        if isinstance(val, Expr):
            return val.node
        if isinstance(val, ExprNode):
            return val
        if isinstance(val, (int, float)):
            return ConstNode(val)
        raise TypeError(f"Unsupported value: {val}")


def sin(expr):
    return Expr(UnaryNode("sin", math.sin, expr.node))


def cos(expr):
    return Expr(UnaryNode("cos", math.cos, expr.node))


def exp(expr):
    return Expr(UnaryNode("exp", math.exp, expr.node))


def log(expr):
    return Expr(UnaryNode("log", math.log, expr.node))


class InputSet:
    def __init__(self):
        self.inputs = {}

    def add(self, name, shape=None, type=float):
        node = VarNode(name, shape=shape, type=type)
        self.inputs[name] = Expr(node)
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared inputs")
        return self.inputs[name]


class ConstantSet:
    def __init__(self):
        self.inputs = {}

    def add(self, name, value=0.0, type=float):
        node = ConstNode(value)
        self.inputs[name] = Expr(node)
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared constants")
        return self.inputs[name]


class VarSet:
    def __init__(self):
        self.vars = {}

    def add(self, name, shape=None, type=float):
        node = VarNode(name, shape=shape, type=type)
        self.vars[name] = Expr(node)
        return

    def __iter__(self):
        return iter(self.vars)

    def __getitem__(self, name):
        return self.vars[name]

    def __setitem__(self, name, expr):
        if name not in self.vars:
            raise KeyError(f"{name} not in declared variables")
        expr.node.name = name
        self.vars[name] = expr


class OutputSet:
    class OutputExpr:
        def __init__(self, name, type=float, shape=None):
            self.name = name
            self.shape = shape
            self.type = type

            if shape is not None and (shape != 1 or shape != (1)):
                if len(shape) == 1:
                    self.expr = [None for _ in range(shape[0])]
                elif len(shape) == 2:
                    self.expr = [
                        [None for _ in range(shape[1])] for _ in range(shape[0])
                    ]
            else:
                self.expr = None

    def __init__(self):
        self.outputs = {}

    def add(self, name, type=float, shape=None):
        self.outputs[name] = self.OutputExpr(name, shape=shape, type=type)
        return

    def __iter__(self):
        return iter(self.outputs)

    def __getitem__(self, name):
        return self.outputs[name]

    def __setitem__(self, name, expr):
        if name not in self.outputs:
            raise KeyError(f"{name} not in declared outputs")

        if isinstance(expr, Expr):
            if self.outputs[name].shape is None:
                self.outputs[name].expr = expr
        else:
            shape = self.outputs[name].shape
            if len(shape) == 1:
                for i in range(shape[0]):
                    self.outputs[name].expr[i] = expr[i]
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.outputs[name].expr[i][j] = expr[i][j]

    def evaluate(self, name, env):
        return self.outputs[name].node.evaluate(env)

    def generate_cpp(self):
        lines = []
        for item in self.outputs:
            if self.outputs[item].shape is None:
                lines.append(
                    f"{self.outputs[item].name} = {self.outputs[item].expr.generate_cpp()}"
                )
            elif len(self.outputs[item].shape) == 1:
                for i in range(self.outputs[item].shape[0]):
                    lines.append(
                        f"{self.outputs[item].name}[{i}] = {self.outputs[item].expr[i].generate_cpp()};"
                    )
            elif len(self.outputs[item].shape) == 2:
                for i in range(self.outputs[item].shape[0]):
                    for j in range(self.outputs[item].shape[1]):
                        lines.append(
                            f"{self.outputs[item].name}({i}, {j}) = {self.outputs[item].expr[i][j].generate_cpp()};"
                        )

        return lines


class Component:
    def __init__(self):
        self.name = self.__class__.__name__
        self.constants = ConstantSet()
        self.data = InputSet()
        self.inputs = InputSet()
        self.vars = VarSet()
        self.outputs = OutputSet()

    def add_constant(self, name, value=1.0, type=float):
        self.constants.add(name, value=value, type=type)

    def add_input(self, name, type=float, shape=None):
        self.inputs.add(name, type=type, shape=shape)

    def add_var(self, name, type=float, shape=None):
        self.vars.add(name, type=type, shape=shape)

    def add_output(self, name, type=float, shape=None):
        self.outputs.add(name, type=type, shape=shape)

    def generate_cpp(self):
        """
        Generate the code for a c++ implementation
        """

        self.compute()
        lines = self.outputs.generate_cpp()

        for line in lines:
            print(line)

        return


class CartComponent(Component):
    def __init__(self, g=9.81, L=0.5, m1=0.5, m2=0.3):
        super().__init__()

        self.add_constant("g", value=g)
        self.add_constant("L", value=L)
        self.add_constant("m1", value=m1)
        self.add_constant("m2", value=m2)

        self.add_input("x")
        self.add_input("q", shape=(4,))
        self.add_input("qdot", shape=(4,))

        self.add_var("cost")
        self.add_var("sint")

        self.add_output("res", shape=(4,))

        return

    def compute(self):
        g = self.constants["g"]
        L = self.constants["L"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]

        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Compute the declared variable values
        self.vars["sint"] = sin(q[1])
        self.vars["cost"] = cos(q[1])

        # Extract a reference to the expressions for convenience
        sint = self.vars["sint"]
        cost = self.vars["cost"]

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * (1.0 - cost * cost)) * qdot[2] - (
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        self.outputs["res"] = res

        return


cart = CartComponent()
cart.generate_cpp()


# class CodeBuilder:
#     def __init__(self):
#         self.inputs = {}
#         self.outputs = {}
#         self.loop_size = None

#     def var(self, name, shape=None):
#         node = VarNode(name, shape=shape)
#         self.inputs[name] = node
#         if shape:
#             self.loop_size = shape[0]
#         return CppExpr(node)

#     def assign(self, name, expr: CppExpr):
#         expr.node.name = name
#         self.outputs[name] = expr.node

#     def evaluate(self, name, env):
#         return self.outputs[name].evaluate(env)

# def generate_cpp(self, result_name, return_type="double"):
#     lines = []
#     size = self.loop_size or 1

#     if self.loop_size:
#         lines.append(f"{return_type} {result_name}[{size}];")
#         lines.append(f"for (int i = 0; i < {size}; ++i) {{")
#         lines.append(f"    {result_name}[i] = {self.outputs[result_name].generate_cpp(index='i')};")
#         lines.append("}")
#         lines.append(f"return 0;")
#     else:
#         lines.append(f"{return_type} {result_name} = {self.outputs[result_name].generate_cpp()};")
#         lines.append(f"return {result_name};")

#     args = ", ".join(f"{return_type} {name}[{self.loop_size}]" if self.inputs[name].shape else f"{return_type} {name}"
#                      for name in self.inputs)
#     return f"{return_type} compute({args}) {{\n    " + "\n    ".join(lines) + "\n}"

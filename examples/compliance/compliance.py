import amigo as am
import numpy as np  # used for plotting/analysis
import argparse


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Filter input values
        self.add_input("x")
        self.add_input("rho")

        self.add_output("rho_res")

        self.add_data("x_coord")
        self.add_data("y_coord")

        self.empty = True

    def compute(self):
        pass


def shape_funcs(xi, eta):
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    return N, Nxi, Neta


def dot(N, u):
    return N[0] * u[0] + N[1] * u[1] + N[2] * u[2] + N[3] * u[3]


def filter_factory(pt: int):
    qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    def init_func(self):
        am.Component.__init__(self)
        self.xi = qpts[pt % 2]
        self.eta = qpts[pt // 2]

        # The filter radius
        self.add_constant("r_filter", 0.1)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/output
        self.add_input("x", shape=(4,))
        self.add_input("rho", shape=(4,))
        self.add_output("rho_res", shape=(4,))

        # Add temporary variables
        self.add_var("detJ")
        self.add_var("invJ", shape=(2, 2))
        self.add_var("Nx", shape=(4))
        self.add_var("Ny", shape=(4))
        self.add_var("rho0")
        self.add_var("x0")
        self.add_var("rho_x")
        self.add_var("rho_y")

        return

    def compute(self):
        r = self.constants["r_filter"]
        x = self.inputs["x"]
        rho = self.inputs["rho"]

        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        N, N_xi, N_ea = shape_funcs(self.xi, self.eta)

        x_xi = dot(N_xi, X)
        x_ea = dot(N_ea, X)

        y_xi = dot(N_xi, Y)
        y_ea = dot(N_ea, Y)

        self.vars["detJ"] = x_xi * y_ea - x_ea * y_xi
        detJ = self.vars["detJ"]

        self.vars["invJ"] = [[y_ea / detJ, -x_ea / detJ], [-y_ea / detJ, x_xi / detJ]]
        invJ = self.vars["invJ"]

        self.vars["Nx"] = [
            invJ[0, 0] * N_xi[0] + invJ[1, 0] * N_ea[0],
            invJ[0, 0] * N_xi[1] + invJ[1, 0] * N_ea[1],
            invJ[0, 0] * N_xi[2] + invJ[1, 0] * N_ea[2],
            invJ[0, 0] * N_xi[3] + invJ[1, 0] * N_ea[3],
        ]

        self.vars["Ny"] = [
            invJ[0, 1] * N_xi[0] + invJ[1, 1] * N_ea[0],
            invJ[0, 1] * N_xi[1] + invJ[1, 1] * N_ea[1],
            invJ[0, 1] * N_xi[2] + invJ[1, 1] * N_ea[2],
            invJ[0, 1] * N_xi[3] + invJ[1, 1] * N_ea[3],
        ]

        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        self.vars["x0"] = dot(N, x)
        self.vars["rho0"] = dot(N, rho)
        self.vars["rho_x"] = dot(Nx, rho)
        self.vars["rho_y"] = dot(Ny, rho)

        x0 = self.vars["x0"]
        rho0 = self.vars["rho0"]
        rho_x = self.vars["rho_x"]
        rho_y = self.vars["rho_y"]

        self.outputs["rho_res"] = [
            detJ * (N[0] * (rho0 - x0) + r * r * (Nx[0] * rho_x + Ny[0] * rho_y)),
            detJ * (N[1] * (rho0 - x0) + r * r * (Nx[1] * rho_x + Ny[1] * rho_y)),
            detJ * (N[2] * (rho0 - x0) + r * r * (Nx[2] * rho_x + Ny[2] * rho_y)),
            detJ * (N[3] * (rho0 - x0) + r * r * (Nx[3] * rho_x + Ny[3] * rho_y)),
        ]

        return

    class_name = f"Filter{pt}"
    return type(
        class_name, (am.Component,), {"__init__": init_func, "compute": compute}
    )()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

nx = 64
ny = 64
nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))

conn = np.zeros((nelems, 4), dtype=int)
for j in range(ny):
    for i in range(nx):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

node_src = NodeSource()

module_name = "compliance"
model = am.Model(module_name)

model.add_component("src", nnodes, node_src)

for n in range(4):
    fltr = filter_factory(n)
    name = f"filter{n}"

    model.add_component(name, nelems, fltr)

    # Link the inputs and the outputs
    model.link(name + ".x_coord", "src.x_coord", tgt_indices=conn)
    model.link(name + ".y_coord", "src.y_coord", tgt_indices=conn)

    model.link(name + ".x", "src.x", tgt_indices=conn)
    model.link(name + ".rho", "src.rho", tgt_indices=conn)
    model.link(name + ".rho_res", "src.rho_res", tgt_indices=conn)

model.initialize()

if args.build:
    model.generate_cpp()
    model.build_module()

print(model.num_variables)
print(model.data_size)

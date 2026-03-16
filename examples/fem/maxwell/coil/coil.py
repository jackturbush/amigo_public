import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import matplotlib.pylab as plt
from examples.fem.maxwell.parser import InpParser
from scipy.sparse.linalg import spsolve
import examples.fem.maxwell.utils as utils
from examples.fem.maxwell.linear_tri_elements import compute_shape_derivs


class Maxwell(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Define constants
        self.add_constant("alpha", 10.0)  # 1/mu_r

        # Define inputs to the problem
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract inputs
        alpha = self.constants["alpha"]
        u = self.inputs["u"]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea, Nx, Ny, detJ, invJ = compute_shape_derivs(xi, eta, X, Y)

        # Compute the local element residual
        K00 = qwts[n] * detJ * alpha * (Nx[0] * Nx[0] + Ny[0] * Ny[0])
        K01 = qwts[n] * detJ * alpha * (Nx[0] * Nx[1] + Ny[0] * Ny[1])
        K02 = qwts[n] * detJ * alpha * (Nx[0] * Nx[2] + Ny[0] * Ny[2])

        K10 = qwts[n] * detJ * alpha * (Nx[1] * Nx[0] + Ny[1] * Ny[0])
        K11 = qwts[n] * detJ * alpha * (Nx[1] * Nx[1] + Ny[1] * Ny[1])
        K12 = qwts[n] * detJ * alpha * (Nx[1] * Nx[2] + Ny[1] * Ny[2])

        K20 = qwts[n] * detJ * alpha * (Nx[2] * Nx[0] + Ny[2] * Ny[0])
        K21 = qwts[n] * detJ * alpha * (Nx[2] * Nx[1] + Ny[2] * Ny[1])
        K22 = qwts[n] * detJ * alpha * (Nx[2] * Nx[2] + Ny[2] * Ny[2])

        res = [
            K00 * u[0] + K01 * u[1] + K02 * u[2],
            K10 * u[0] + K11 * u[1] + K12 * u[2],
            K20 * u[0] + K21 * u[1] + K22 * u[2],
        ]

        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]


class Coil(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Constants
        self.add_constant("Jz", value=5.0)

        # Add input
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        # Objective
        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Extract inputs
        u = self.inputs["u"]

        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea, Nx, Ny, detJ, invJ = compute_shape_derivs(xi, eta, X, Y)
        Jz = self.constants["Jz"]

        res = [
            -1 * qwts[n] * detJ * Jz * N[0],
            -1 * qwts[n] * detJ * Jz * N[1],
            -1 * qwts[n] * detJ * Jz * N[2],
        ]
        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]
        return


class DirichletBc(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("u", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        self.objective["obj"] = self.inputs["u"] * self.inputs["lam"]
        return


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")
        self.add_data("y_coord")

        # States
        self.add_input("u")


if __name__ == "__main__":
    # Retrieve mesh information for the analysis
    inp_filename = "plate.inp"
    parser = InpParser()
    parser.parse_inp(inp_filename)

    # Get the node locations
    X = parser.get_nodes()

    # Get element connectivity
    conn = parser.get_conn("SURFACE1", "CPS3")

    # Get the boundary condition nodes
    edge1 = parser.get_conn("LINE1", "T3D2")
    edge2 = parser.get_conn("LINE2", "T3D2")
    edge3 = parser.get_conn("LINE3", "T3D2")
    edge4 = parser.get_conn("LINE4", "T3D2")

    # Concatenate the unique node tgas for the dirichlet bc
    dirichlet_bc_tags = np.concatenate(
        (
            edge1.flatten(),
            edge2.flatten(),
            # edge3.flatten(),
            # edge4.flatten(),
        ),
        axis=None,
    )
    dirichlet_bc_tags = np.unique(dirichlet_bc_tags, sorted=True)

    # Define the total number of elements and nodes in the mesh
    nelems = conn.shape[0]
    nnodes = X.shape[0]

    # Define parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )

    args = parser.parse_args()

    # Define your amigo module
    module_name = "maxwell"
    model = am.Model(module_name=module_name)

    # Add physics components to amigo model
    maxwell = Maxwell()
    model.add_component(name="maxwell", size=nelems, comp_obj=maxwell)

    coils = Coil()
    model.add_component(name="coils", size=nelems, comp_obj=coils)

    dirichlet_bc = DirichletBc()
    model.add_component(
        "dirichlet_bc", size=len(dirichlet_bc_tags), comp_obj=dirichlet_bc
    )
    model.link("src.u", "dirichlet_bc.u", src_indices=dirichlet_bc_tags)

    # Add source components to the amigo model
    node_src = NodeSource()
    model.add_component("src", nnodes, node_src)

    # Ex: maxwell.y_coord = src.y_coord[conn]
    model.link("maxwell.x_coord", "src.x_coord", tgt_indices=conn)
    model.link("maxwell.y_coord", "src.y_coord", tgt_indices=conn)
    model.link("coils.x_coord", "src.x_coord", tgt_indices=conn)
    model.link("coils.y_coord", "src.y_coord", tgt_indices=conn)
    model.link("coils.u", "src.u", tgt_indices=conn)
    model.link("maxwell.u", "src.u", tgt_indices=conn)

    # Build module
    if args.build:
        model.build_module()

    # Initialize the model
    model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
    p = model.get_problem()

    # Set the problem data
    data = model.get_data_vector()
    data["src.x_coord"] = X[:, 0]
    data["src.y_coord"] = X[:, 1]

    # Vectors for solving the problem
    mat = p.create_matrix()
    alpha = 1.0
    x = p.create_vector()
    ans = p.create_vector()
    g = p.create_vector()
    rhs = p.create_vector()
    p.hessian(alpha, x, mat)
    p.gradient(alpha, x, g)
    csr_mat = am.tocsr(mat)

    # Plot solution field
    ans.get_array()[:] = spsolve(csr_mat, g.get_array())
    ans_local = ans
    vals = ans_local.get_array()[model.get_indices("src.u")]
    utils.plot_solution(X, conn, vals, title="coil")
    plt.show()

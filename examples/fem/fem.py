import amigo as am
import numpy as np
import re
import basis
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from connectivity import InpParser
from matplotlib.collections import PolyCollection


class DofSource(am.Component):
    def __init__(self, input_names=[], geo_names=[], data_names=[], con_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)
        for name in geo_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)

        return


class DegreesOfFreedom:
    def __init__(self, mesh, space, kind="input"):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.space = space
        self.kind = kind

        return

    def _initialize(self):
        """
        Initialize the degrees of freedom associated with this mesh
        """

        # Allocate degrees of freedom for each of the nodes/

        return

    def add_source(self, model):

        # if "H1" in self.space.get_spaces():
        #     model.add_component(f"src_{self.kind}", ndof, )
        # if "H1" in self.space.get_spaces():

        # if "H1" in self.space.get_spaces():

        pass

    def get_basis(self, etype, space, names=[], kind="input"):
        if etype == "CPS3":
            if space == "H1":
                return basis.TriangleLagrangeBasis(1, names, kind=kind)
        elif etype == "CPS4":
            if space == "H1":
                return basis.QuadLagrangeBasis(1, names, kind=kind)
        elif etype == "CPS6":
            if space == "H1":
                return basis.TriangleLagrangeBasis(2, names, kind=kind)
        elif etype == "M3D9":
            if space == "H1":
                return basis.QuadLagrangeBasis(2, names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CPS3":
            return basis.TriangleQuadrature(2)
        elif etype == "CPS4":
            return basis.QuadQuadrature(2)
        elif etype == "CPS6":
            return basis.TriangleQuadrature(4)
        elif etype == "M3D9":
            return basis.QuadQuadrature(3)

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")

    def link_dof(self, model, name, elem_name, conn):
        model.link(f"src.{name}", f"{elem_name}.{name}", src_indices=conn)
        return


class Mesh:
    def __init__(self, filename):
        self.parser = InpParser()
        self.parser.parse_inp(filename)

        self.X = self.parser.get_nodes()

    def get_num_nodes(self):
        return self.X.shape[0]

    def get_domains(self):
        domains = self.parser.get_domains()

        element_types = ["CPS3", "CPS4", "CPS6", "M3D9"]

        volumes = {}
        for name in domains:
            for etype in element_types:
                if etype in domains[name]:
                    volumes[name] = domains[name]
                    break

        return volumes

    def get_conn(self, name, etype):
        return self.parser.get_conn(name, etype)

    def plot(self, u, ax=None, nlevels=30, cmap="coolwarm"):
        min_level = np.min(u)
        max_level = np.max(u)
        levels = np.linspace(min_level, max_level, nlevels)

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        volumes = self.get_domains()
        x = self.X[:, 0]
        y = self.X[:, 1]

        for name in volumes:
            for etype in volumes[name]:
                # Get the connectivity
                conn = self.convert_conn(etype, self.get_conn(name, etype))
                tri = mtri.Triangulation(x, y, conn)

                # Set the contour plot
                ax.tricontourf(tri, u, levels=levels, cmap=cmap)
                ax.tricontour(
                    tri, u, levels=levels, colors="k", linewidths=0.3, alpha=0.5
                )

                # Overlay the mesh skeleton
                gmsh_conn = self.get_conn(name, etype)
                X2d = self.X[:, 0:2]
                polygons = [X2d[row] for row in gmsh_conn]
                mesh = PolyCollection(
                    polygons,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.4,
                )
                ax.add_collection(mesh)
                ax.set_aspect("equal")
        return ax

    def convert_conn(self, etype, conn):
        if etype == "CPS3":
            return conn
        elif etype == "CPS4":
            c = [[0, 1, 2], [0, 2, 3]]
        elif etype == "CPS6":
            # 2
            # |  .
            # 5     4
            # |        .
            # 0 --- 3 --- 1
            c = [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]]
        elif etype == "M3D9":
            # 3 --- 6 --- 2
            # |           |
            # 7     8     5
            # |           |
            # 0 --- 4 --- 1
            c = [
                [0, 4, 7],
                [4, 8, 7],
                [4, 1, 8],
                [1, 5, 8],
                [7, 8, 3],
                [8, 6, 3],
                [8, 5, 6],
                [5, 2, 6],
            ]

        cs = []
        for c0 in c:
            cs.append(conn[:, c0])

        return np.vstack(cs)


class Problem:
    # soln_space = object
    def __init__(self, mesh, soln_space, weakform, data_space=[], geo_space=[], ndim=2):

        self.mesh = mesh
        self.ndim = ndim  # Dimension of the problem

        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        # """Tell the mesh what dof, and basis"""
        # if self.ndim == 2:
        #     self.geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

        self.weakform = weakform

        # Initialize Dof's
        self.soln_dof = DegreesOfFreedom(mesh, "H1", "soln")
        self.geo_dof = DegreesOfFreedom(mesh, "H1", "geo")
        self.data_dof = DegreesOfFreedom(mesh, "H1", "data")

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        # Get the names of things associated with H1
        input_names = self.soln_space.get_names("H1")  # rho
        data_names = self.data_space.get_names("H1")  # x, y
        geo_names = self.geo_space.get_names("H1")  # x, y

        # Create amigo component with input names and geo names
        self.dof_src = DofSource(input_names=input_names, geo_names=geo_names)

        # Add global mesh source component
        nnodes = mesh.get_num_nodes()
        model.add_component("src", nnodes, self.dof_src)

        # Build the elements for all domains
        domains = self.mesh.get_domains()
        for domain in domains:
            for etype in domains[domain]:
                # Build a finite-element for each weak form
                elem_name = f"Element{etype}_{domain}"

                soln_basis = self.soln_dof.get_basis(
                    etype, "H1", names=input_names, kind="input"
                )
                data_basis = self.data_dof.get_basis(
                    etype, "H1", names=data_names, kind="data"
                )
                geo_basis = self.geo_dof.get_basis(
                    etype, "H1", names=geo_names, kind="data"
                )

                # Create the quadrature instance
                quadrature = self.soln_dof.get_quadrature(etype)

                # Create the element object
                elem = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    self.weakform,
                    etype,
                    input_names,
                    data_names,
                    geo_names,
                )

                # Get the connectivity
                # Needs to pull out any type of connectivity for the basis
                conn = self.mesh.get_conn(domain, etype)

                # Add the element/component
                nelems = conn.shape[0]
                model.add_component(elem_name, nelems, elem)

                # Link all the element dof to the component
                for name in input_names:
                    self.soln_dof.link_dof(model, name, elem_name, conn)

                for name in geo_names:
                    self.geo_dof.link_dof(model, name, elem_name, conn)

        return model


class FiniteElement(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        weakform,
        etype,
        input_names=[],
        data_names=[],
        geo_names=[],
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.weakform = weakform

        # Add the declarations for each basis used in the element
        # Declrations = add inputs, add data etc.
        # self.soln_basis.add_declarations(self)
        # self.data_basis.add_declarations(self)
        # self.geo_basis.add_declarations(self)

        # The x/y coordinates
        if etype == "CPS3":
            shape = (3,)

        elif etype == "CPS4":
            shape = (4,)

        # Data
        for name in geo_names:
            self.add_data(name, shape=shape)

        # Inputs
        for name in input_names:
            self.add_input(name, shape=shape)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        self.add_objective("obj")

        return

    def compute(self, **args):

        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates
        detJ, Jinv = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(detJ, Jinv, soln_xi)
        data_phys = self.data_basis.transform(detJ, Jinv, data_xi)

        # Add the contributions directly to the Lagrangian
        self.objective["obj"] = (
            quad_weight * detJ * self.weakform(soln_phys, data=data_phys, geo=geo)
        )

        return


def weakform(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    # u = soln["u"]
    # uvalue = u["value"]
    # ugrad = u["grad"]

    # v = soln["v"]
    # vvalue = v["value"]
    # vcurl = v["curl"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]
    # rho = data["rho"]["value"]

    f = am.sin(x) ** 2 * am.cos(y) ** 2
    return 0.5 * (uvalue**2 + basis.dot_product(ugrad, ugrad, n=2) - 2.0 * uvalue * f)


soln_space = basis.SolutionSpace({"u": "H1"})
data_space = basis.SolutionSpace({"x": "H1", "y": "H1"})
geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

mesh = Mesh("magnet_order_1.inp")
# mesh = Mesh("plate.inp")
problem = Problem(
    mesh,
    soln_space,
    weakform,
    data_space=data_space,
    geo_space=geo_space,
    ndim=2,
)

model = problem.create_model("test")

model.build_module()
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

# print("num_variables = ", model.num_variables)
problem = model.get_problem()

# Set the problem data
data = model.get_data_vector()
data["src.x"] = mesh.X[:, 0]
data["src.y"] = mesh.X[:, 1]

x = problem.create_vector()
mat = problem.create_matrix()
rhs = model.create_vector()
problem.gradient(1.0, x, rhs.get_vector())
problem.hessian(1.0, x, mat)

chol = am.SparseCholesky(mat)
flag = chol.factor()
print("flag = ", flag)
chol.solve(rhs.get_vector())

u = rhs["src.u"]
mesh.plot(u)
plt.show()

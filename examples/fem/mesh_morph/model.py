import numpy as np
from amigo.fem import Mesh, Problem, SolutionSpace
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse
import matplotlib.tri as tri


def plot_mesh(mesh, dx, dy):
    fig, ax = plt.subplots()

    # Get the domains in the mesh
    domains = mesh.get_domains()
    etypes = ["CPS3"]
    for name in domains:
        for etype in domains[name]:
            if etype not in etypes:
                continue
            conn = mesh.get_conn(name, etype)
            triang = tri.Triangulation(mesh.X[:, 0], mesh.X[:, 1], conn)
            triang_deformed = tri.Triangulation(
                mesh.X[:, 0] + dx, mesh.X[:, 1] + dy, conn
            )
            ax.triplot(triang, color="grey", linestyle="--", linewidth=1.0)
            ax.triplot(triang_deformed, color="blue", linestyle="-", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_axis_off()
    return


def strain_energy_integrand(soln, data=None, geo=None):
    # Gradients of the solution field
    dx_grad = soln["dx"]["grad"]
    dy_grad = soln["dy"]["grad"]

    # geometry vars
    x = geo["x"]["value"]
    y = geo["y"]["value"]

    # Strains
    exx = dx_grad[0]
    eyy = dy_grad[1]
    exy = dx_grad[1] + dy_grad[0]

    # Constitutive model
    E = 1.0
    nu = 0.3
    alpha = E / (1 - nu**2)

    # Assume uniform thickness
    t = 1.0

    C11 = 1.0
    C12 = nu
    C13 = 0.0
    C21 = nu
    C22 = 1.0
    C23 = 0.0
    C31 = 0.0
    C32 = 0.0
    C33 = 0.5 * (1 - nu)

    # Constant strain elements total internal strain energy
    Wx = exx * (C11 * exx + C12 * eyy + C13 * exy)
    Wy = eyy * (C21 * exx + C22 * eyy + C23 * exy)
    Wxy = exy * (C31 * exx + C32 * eyy + C33 * exy)

    # Total strain potential energy integrand
    W = 0.5 * alpha * (Wx + Wy + Wxy) * t

    # Alternative formulation for the total strain energy integrad
    # W = 0.5 * (exx**2 + eyy**2 + 2.0 * nu * exx * eyy + 0.5 * (1.0 - nu) * exy**2)

    return W


# Define the spaces for the solutions
soln_space = SolutionSpace({"dx": "H1", "dy": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})

integrand_map = {
    "plane_stress": {
        "target": ["SURFACE1", "SURFACE2"],
        "integrand": strain_energy_integrand,
    },
}

bc_map = {
    "clamp": {
        "type": "dirichlet",
        "target": ["LINE3"],
        "input": ["dx", "dy"],
    },
    "move": {
        "type": "dirichlet",
        "target": ["LINE1"],
        "input": ["dx", "dy"],
    },
    "move_y": {
        "type": "dirichlet",
        "target": ["LINE2", "LINE4"],
        "input": ["dy"],
    },
}

# Create a global model
model = am.Model("model")

# Initialize the mesh object
mesh = Mesh("mesh.inp")

# Parser agrs
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Create the submodel for the plane stress problem
problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)
submodel = problem.create_model("mesh_morph")

# Add the submodel to the global model
model.add_model("submodel", submodel)

if args.build:
    model.build_module()

model.initialize()

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

# An approach for enforced dirichlet bcs
nodes = mesh.get_nodes_in_domain("LINE1")


for i in nodes:
    y_coord = mesh.X[i][1]
    x[f"submodel.soln.dx[{i}]"] = -np.cos(np.pi * y_coord / 5)
    # x[f"submodel.soln.dy[{i}]"] = 5.0

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

csr_mat = am.tocsr(mat)
x[:] -= spsolve(csr_mat, g[:])

# Extract displacement fields
dx = x["submodel.soln.dx"]
dy = x["submodel.soln.dy"]

# Plot the mesh
plot_mesh(mesh, dx, dy)
plt.show()

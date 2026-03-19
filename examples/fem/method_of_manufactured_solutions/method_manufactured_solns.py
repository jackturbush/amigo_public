import numpy as np
import amigo as am
from amigo.fem import dot_product, SolutionSpace, Problem, Mesh
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse

"""
Method of Manufactured Solutions
    u = sin(2πx)sin(2πy)
    ∇·∇u - f = 0
    f=(8π**2)sin(2xπ)sin(2yπ)
"""


def output(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    return {"integral": uvalue}


def potential(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    pi = np.pi
    f = 8 * pi**2 * am.sin(2 * x * pi) * am.sin(2 * y * pi)

    wf = 0.5 * dot_product(ugrad, ugrad, n=2) + f * uvalue
    return wf


def exact(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

meshes = {"Mesh": Mesh("plate.inp")}
bc_map = {
    "Mesh": {
        "DirichletLine1": {
            "type": "dirichlet",
            "target": [
                "LINE1",
                "LINE2",
                "LINE3",
                "LINE4",
            ],
            "input": ["u"],
        },
    },
}

# Weak form mapping for each mesh
potential_map = {
    "Mesh": {
        "physics": {
            "target": ["SURFACE1"],
            "potential": potential,
        },
    }
}

output_map = {
    "Mesh": {
        "integral": {
            "names": ["integral"],
            "target": ["SURFACE1"],
            "function": output,
        },
    },
}

# Initialize the spaces (same for all domains)
soln_space = SolutionSpace({"u": "H1"})
data_space = SolutionSpace({})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})

# Define the mesh object
mesh = meshes["Mesh"]

# Define the global amigo model
model = am.Model("mms_module")

# Create an amigo model for each mesh
for mesh_name, mesh in meshes.items():
    problem = Problem(
        mesh,
        soln_space,
        data_space,
        geo_space,
        potential_map=potential_map[mesh_name],
        bc_map=bc_map[mesh_name],
        output_map=output_map[mesh_name],
    )
    sub_model = problem.create_model(mesh_name)
    model.add_model(mesh_name, sub_model)

# Build the model
if args.build:
    model.build_module()
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

# Set the problem data
data = model.get_data_vector()
data["Mesh.src_geo.x"] = mesh.X[:, 0]
data["Mesh.src_geo.y"] = mesh.X[:, 1]

mat = p.create_matrix()
alpha = 1.0
x = p.create_vector()
ans = p.create_vector()
g = p.create_vector()
rhs = p.create_vector()
p.hessian(alpha, x, mat)
p.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
u = ans_local.get_array()[model.get_indices("Mesh.src_soln.u")]
u_exact = exact(x=mesh.X[:, 0], y=mesh.X[:, 1])

# Get the output
output = model.create_output_vector()
p.compute_output(ans, output.get_vector())
print("The integral is: ", output["Mesh.outputs.integral[0]"])

# Plot the solution field
mesh.plot(u)
plt.show()

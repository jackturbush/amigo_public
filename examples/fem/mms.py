import numpy as np
import amigo as am
from amigo.fem import dot_product, SolutionSpace, Problem, Mesh

from pathlib import Path
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

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

    pi = np.pi
    integrand = am.sin(2 * pi * x) * am.cos(2 * pi * y)
    # return {"integrand": integrand}
    return {"torque": uvalue}


def weakform(soln, data=None, geo=None):
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


# Loop through each mesh refinemens and store the 2 norm
n = 20

norms = np.zeros(n)
number_elems = np.zeros(n)
# lc_vals = np.linspace(1e-1, 4e-3, n)
lc_vals = [8e-3]

recombine = False
order = 1
if order == 1:
    elem_type = "CPS3"
elif order == 2:
    elem_type = "CPS6"

for i, lc in enumerate(lc_vals):
    meshes = {"Mesh": Mesh("plate.inp")}
    dirichlet_bc_meshes = {
        "Mesh": {
            "DirichletLine1": {
                "type": "dirichlet",
                "target": "LINE1",
                "input": ["u"],
                "start": True,
                "end": True,
            },
            "DirichletLine2": {
                "type": "dirichlet",
                "target": "LINE2",
                "input": ["u"],
                "start": False,
                "end": False,
            },
            "DirichletLine3": {
                "type": "dirichlet",
                "target": "LINE3",
                "input": ["u"],
                "start": True,
                "end": True,
            },
            "DirichletLine4": {
                "type": "dirichlet",
                "target": "LINE4",
                "input": ["u"],
                "start": False,
                "end": False,
            },
        },
    }

    # Symmetric BCs mapping for each mesh
    symm_bc_meshes = {"Mesh": {}}

    # Weak form mapping for each mesh
    weakform_map = {
        "mms": {
            "target": ["SURFACE1"],
            "weakform": weakform,
        }
    }

    wf_mesh_map = {
        "Mesh": weakform_map,
    }

    output_map = {
        "integral": {"names": ["torque"], "target": ["SURFACE1"], "function": output},
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
            weakform_map=wf_mesh_map[mesh_name],
            dirichlet_bc_map=dirichlet_bc_meshes[mesh_name],
            output_map=output_map,
        )
        sub_model = problem.create_model(mesh_name)
        model.add_model(mesh_name, sub_model)

    # Build the model
    # source_dir = Path(__file__).resolve().parent
    model.build_module()  # source_dir=source_dir)
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

    output = model.create_output_vector()
    p.compute_output(ans, output.get_vector())

    print("The torque is: ", output["Mesh.outputs.torque[0]"])

    mesh.plot(u)

    delta_u = u - u_exact
    norm = np.linalg.norm(delta_u)
    print(f"\n||err||2 = {norm:4e}")
    norms[i] = norm

    plt.show()

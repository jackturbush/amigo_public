import gmsh
import sys


def check_areas(X, conn):
    # Reshape arrays
    conn = conn.reshape(-1, 3)
    X = X.reshape(-1, 3)

    # Check element area
    for i in range(conn.shape[0]):
        # Zero indexing required
        n1 = conn[i, 0] - 1
        n2 = conn[i, 1] - 1
        n3 = conn[i, 2] - 1

        n1_x = X[n1, 0]
        n1_y = X[n1, 1]

        n2_x = X[n2, 0]
        n2_y = X[n2, 1]

        n3_x = X[n3, 0]
        n3_y = X[n3, 1]

        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        if a_e <= 0.0:
            raise Exception(f"Element area for element {i} = {a_e}")


gmsh.initialize()
gmsh.model.add("multiple_weakform_test")
lc = 4e-1
lc1 = 4e-2

# Domain 1 geometry
l = 5.0  # length
w = 5.0  # width

# Geometry for sub domain 2
xc2 = 1  # bottom left point offset in x
yc2 = 2.5  # bottom left point offset in y
l2 = 1.0  # length
w2 = 1.0  # width

# Geometry for sub domain 3
xc3 = 3.5  # bottom left point offset in x
yc3 = 2.5  # bottom left point offset in y
l3 = 1.0  # length
w3 = 1.0  # width

# Add points
geom = gmsh.model.geo

# Main Domain
geom.addPoint(0, 0, 0, lc, 1)
geom.addPoint(l, 0, 0, lc, 2)
geom.addPoint(l, w, 0, lc, 3)
geom.addPoint(0, w, 0, lc, 4)

# Internal Domain 1
geom.addPoint(xc2, yc2, 0, lc1, 5)
geom.addPoint(l2 + xc2, yc2, 0, lc1, 6)
geom.addPoint(l2 + xc2, w2 + yc2, 0, lc1, 7)
geom.addPoint(xc2, w2 + yc2, 0, lc1, 8)

# Internal Domain 2
geom.addPoint(xc3, yc3, 0, lc1, 9)
geom.addPoint(l3 + xc3, yc3, 0, lc1, 10)
geom.addPoint(l3 + xc3, w3 + yc3, 0, lc1, 11)
geom.addPoint(xc3, w3 + yc3, 0, lc1, 12)

# Define lines for the boundary loop
geom.addLine(1, 2, 1)
geom.addLine(2, 3, 2)
geom.addLine(3, 4, 3)
geom.addLine(4, 1, 4)

# Define lines for the subdomain 1 loop
geom.addLine(5, 6, 5)
geom.addLine(6, 7, 6)
geom.addLine(7, 8, 7)
geom.addLine(8, 5, 8)

# Define lines for the subdomain 2 loop
geom.addLine(9, 10, 9)
geom.addLine(10, 11, 10)
geom.addLine(11, 12, 11)
geom.addLine(12, 9, 12)

# Define curve loops
geom.addCurveLoop([1, 2, 3, 4], 1, reorient=False)
geom.addCurveLoop([5, 6, 7, 8], 2, reorient=False)
geom.addCurveLoop([9, 10, 11, 12], 3, reorient=False)

# Define surfaces
geom.addPlaneSurface([1, 2, 3], 1)
geom.addPlaneSurface([2], 2)
geom.addPlaneSurface([3], 3)

# Required to call synchronize in order to be meshed
gmsh.model.geo.synchronize()

# Defin physical groups
# gmsh.model.addPhysicalGroup(1, [1], name="Shared Interface")
# gmsh.model.addPhysicalGroup(1, [4], name="PBC Left Edge")
# gmsh.model.addPhysicalGroup(1, [2], name="PBC Right Edge")
# gmsh.model.addPhysicalGroup(1, [3], name="Dirichlet BC Edge")
# gmsh.model.addPhysicalGroup(2, [1], name="Domain 1")
# gmsh.model.addPhysicalGroup(2, [2], name="Domain 2")
# gmsh.model.addPhysicalGroup(2, [3], name="Domain 3")

# Generate 2d mesh
order = 1
gmsh.model.mesh.setRecombine(2, 2)
gmsh.model.mesh.setRecombine(2, 3)
gmsh.model.mesh.setTransfiniteCurve(1, 20)
gmsh.model.mesh.setTransfiniteCurve(2, 20)
gmsh.model.mesh.setTransfiniteCurve(3, 20)
gmsh.model.mesh.setTransfiniteCurve(4, 20)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(order)  # set the order

# Save the mesh
gmsh.write(f"weakform_test_mesh.inp")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

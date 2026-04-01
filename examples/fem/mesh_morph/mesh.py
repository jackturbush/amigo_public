import gmsh
import sys

gmsh.initialize()
gmsh.model.add("mesh")

# Mesh refinement at nodes
lc = 10e-1
lc1 = 1e-1

# Geometry dimentions
hb = 5  # Boundary length and width
hm = 1  # Magnet length and width

# Add points
geom = gmsh.model.geo
geom.addPoint(hb, -hb, 0, lc1, 1)
geom.addPoint(hb, hb, 0, lc1, 2)
geom.addPoint(-hb, hb, 0, lc, 3)
geom.addPoint(-hb, -hb, 0, lc, 4)
geom.addPoint(hm, -hm, 0, lc, 5)
geom.addPoint(hm, hm, 0, lc, 6)
geom.addPoint(-hm, hm, 0, lc, 7)
geom.addPoint(-hm, -hm, 0, lc, 8)

# Define lines for the boundary loop
geom.addLine(1, 2, 1)
geom.addLine(2, 3, 2)
geom.addLine(3, 4, 3)
geom.addLine(4, 1, 4)

# Define lines for the magnet loop
geom.addLine(5, 6, 5)
geom.addLine(6, 7, 6)
geom.addLine(7, 8, 7)
geom.addLine(8, 5, 8)

# Define curve loops
geom.addCurveLoop([1, 2, 3, 4], 1, reorient=False)
geom.addCurveLoop([5, 6, 7, 8], 2, reorient=False)

# Define surfaces
geom.addPlaneSurface([1, 2], 1)
geom.addPlaneSurface([2], 2)

# Required to call synchronize in order to be meshed
gmsh.model.geo.synchronize()

# Generate 2d mesh
# order = 1
# gmsh.model.mesh.setRecombine(2, 1)
# gmsh.model.mesh.setRecombine(2, 2)
gmsh.model.mesh.generate(2)
# gmsh.model.mesh.setOrder(order)  # set the order

# Save the mesh
gmsh.write(f"mesh.inp")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

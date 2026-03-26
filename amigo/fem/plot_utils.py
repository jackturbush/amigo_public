import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def _convert_conn(etype, conn):
    c = None
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

    if c is None:
        raise ValueError(f"Cannot convert mesh for {etype}")

    cs = []
    for c0 in c:
        cs.append(conn[:, c0])

    return np.vstack(cs)


def plot_mesh(X, elem_conn, edge_dict, jpg_name=None):
    fig, ax = plt.subplots()

    # --- Plot element triangulation (background mesh) ---
    triang = tri.Triangulation(X[:, 0], X[:, 1], elem_conn)
    ax.triplot(triang, color="grey", linestyle="--", linewidth=1.0)

    # --- Plot node tags ---
    for n in range(len(X)):
        ax.text(X[n, 0], X[n, 1], f"{n}", fontsize=8)

    # --- Plot unique global edges ---
    for edge_tag, nodes in edge_dict.items():
        if len(nodes) == 2:
            # Linear edge
            n1, n2 = nodes
            x = [X[n1, 0], X[n2, 0]]
            y = [X[n1, 1], X[n2, 1]]
            ax.plot(x, y, linewidth=2.0)

            xm = 0.5 * (x[0] + x[1])
            ym = 0.5 * (y[0] + y[1])
        else:
            raise ValueError("nnodes along edge > 2")

        # Edge label
        ax.text(xm, ym, f"E{edge_tag}", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if jpg_name is not None:
        plt.savefig(jpg_name + ".jpg", dpi=1000)

    return


def plot(
    mesh,
    u,
    ax=None,
    nlevels=30,
    cmap="coolwarm",
    title=None,
    x_offset=0.0,
    y_offset=0.0,
    min_level=None,
    max_level=None,
):
    if min_level == None or max_level == None:
        min_level = np.min(u)
        max_level = np.max(u)

    levels = np.linspace(min_level, max_level, nlevels)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    domains = mesh.get_domains()
    x = mesh.X[:, 0] + x_offset
    y = mesh.X[:, 1] + y_offset

    # We only plot these elements for now
    element_types = ["CPS3", "CPS4", "CPS6", "M3D9"]

    for name in domains:
        for etype in domains[name]:
            if not (etype in element_types):
                continue

            # Get the connectivity
            orig_conn = mesh.get_conn(name, etype)
            conn = _convert_conn(etype, orig_conn)
            triangles = tri.Triangulation(x, y, conn)

            # Set the contour plot
            ax.tricontourf(triangles, u, levels=levels, cmap=cmap)
            ax.tricontour(
                triangles, u, levels=levels, colors="k", linewidths=0.3, alpha=0.5
            )

            if title is not None:
                ax.set_title(title)

            ax.set_aspect("equal")

    return ax

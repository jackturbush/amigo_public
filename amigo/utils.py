import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import BSpline
from .component import Component

try:
    from petsc4py import PETSc
except:
    PETSc = None


def tocsr(mat):
    nrows, ncols, nnz, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()
    return csr_matrix((data, cols, rowp), shape=(nrows, ncols))


def topetsc(mat):
    if PETSc is None:
        return None

    # Extract the data from the matrix
    _, ncols, _, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()

    row_owners = mat.get_row_owners()
    col_owners = mat.get_column_owners()

    comm = row_owners.get_mpi_comm()
    nrows_local = row_owners.get_local_size()
    ncols_local = col_owners.get_local_size()

    A = PETSc.Mat().create(comm=comm)

    sr = (nrows_local, None)
    sc = (ncols_local, ncols)
    A.setSizes((sr, sc), bsize=1)
    A.setType(PETSc.Mat.Type.MPIAIJ)

    nnz_local = rowp[nrows_local]
    A.setValuesCSR(rowp[: nrows_local + 1], cols[:nnz_local], data[:nnz_local])
    A.assemble()

    return A


class BSplineInterpolant(Component):
    def __init__(self, k: int = 4, n: int = 10):
        """
        Perform a BSpline interpolation of evenly spaced input data to output data

        res = output - N(xi) * input

        Args:
            k (int) : Order of the bspline polynomial (degree + 1)
            n (int) : Number of interpolating points
        """
        super().__init__()

        # Set the order of the bspline
        self.k = k

        # Set the number of inputs
        self.n = n

        # Set the size of the input data
        self.add_data("N", shape=self.k)

        # Set the input values to be interpolated
        self.add_input("input", shape=self.k)

        # Set the output values
        self.add_input("output")

        # Set the coupling constraint
        self.add_constraint("res")

        return

    def compute(self):
        N = self.data["N"]
        input = self.inputs["input"]
        output = self.inputs["output"]

        value = 0.0
        for i in range(self.k):
            value = value + N[i] * input[i]
        self.constraints["res"] = output - value

        return

    def compute_knots(self):
        """
        Compute the BSpline knots for interpolation
        """

        # Set the knot locations
        t = np.zeros(self.n + self.k)
        t[: self.k] = 0.0
        t[-self.k : :] = 1.0
        t[self.k - 1 : -self.k + 1] = np.linspace(0, 1, self.n - self.k + 2)

        return t

    def compute_basis(self, xi):
        """
        Compute the BSpline basis functions at the given knot locations
        """

        t = self.compute_knots()

        # Special case for right endpoint
        xi_clamped = xi.copy()
        xi_clamped[xi_clamped == t[-1]] = np.nextafter(t[-1], -np.inf)

        # Find span index i such that t[i] <= x < t[i+1]
        span = np.searchsorted(t, xi_clamped, side="right") - 1
        span = np.clip(span, self.k - 1, self.n - 1)

        N = np.zeros((xi.size, self.k), dtype=float)

        # Evaluate using the Cox–de Boor recursion for each point
        for j in range(xi.size):
            i = span[j]
            # zeroth-degree basis
            N_j = np.zeros(self.k)
            N_j[0] = 1.0

            for d in range(1, self.k):
                saved = 0.0
                for r in range(d):
                    left = t[i - d + 1 + r]
                    right = t[i + 1 + r]
                    denom = right - left
                    temp = 0.0 if denom == 0.0 else N_j[r] / denom
                    N_j[r] = saved + (right - xi_clamped[j]) * temp
                    saved = (xi_clamped[j] - left) * temp
                N_j[d] = saved
            N[j, :] = N_j

        # Handle x == t[-1]
        last_mask = xi == t[-1]
        if np.any(last_mask):
            N[last_mask, :] = 0.0
            N[last_mask, -1] = 1.0

        return N

    def set_data(self, name, npts, data):
        """
        Set the interpolation data for the component group
        """

        # Set the locations for where to evaluate the bspline basis
        xi = np.linspace(0, 1, npts)
        data[f"{name}.N"] = self.compute_basis(xi)

        return

    def add_links(self, name, npts, model, src_name):
        """
        Add the links to the model for the interpolation
        """

        # Set the locations for where to evaluate the bspline basis points
        xi = np.linspace(0, 1, npts)

        # Set the knot locations
        t = self.compute_knots()

        index = self.k - 1
        for i in range(npts):
            while index < self.n and xi[i] > t[index + 1]:
                index += 1

            for j in range(self.k):
                src = f"{src_name}[0, {index - self.k + 1 + j}]"
                target = f"{name}.input[{i}, {j}]"
                model.link(src, target)

        # Build via canonical coefficients
        N = []
        for i in range(self.n):
            c = np.zeros(self.n)
            c[i] = 1.0
            Bi = BSpline(t, c, self.k - 1, extrapolate=False)
            N.append(Bi(xi))

        return

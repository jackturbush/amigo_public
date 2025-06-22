import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .amigo import InteriorPointOptimizer

class Optimizer:
    def __init__(self, model, prob, x, lower, upper, options={}):
        self.model = model
        self.prob = prob
        self.x = x
        self.lower = lower
        self.upper = upper

        # Create the interior point optimizer object
        self.optimizer = InteriorPointOptimizer(self.prob, lower, upper)

        # Create data that will be used in conjunction with the optimizer
        self.vars = self.optimizer.create_opt_vector(self.x)
        self.res = self.optimizer.create_opt_vector()
        self.update = self.optimizer.create_opt_vector()

        # Create vectors that store problem-specific information
        self.grad = self.prob.create_vector()
        self.px = self.prob.create_vector()
        self.bx = self.prob.create_vector()

        # Create hessian matrix object
        self.hess = self.prob.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

    def _get_scipy_csr_mat(self):
        data = self.hess.get_data()
        return csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

    def _solve(self, hess, b, p):
        """
        Solve the KKT syste - many different appraoches could be inserted here
        """

        H = self._get_scipy_csr_mat()

        # Compute the solution using scipy
        self.px.get_array()[:] = spsolve(H, self.bx.get_array())

        return
    
    def optimize(self):

        barrier_param = 1.0
        tau = 0.95

        for i in range(3):
            print("Iteration ", i)
            # Compute the gradient and the Hessian matrix
            self.prob.gradient(self.x, self.grad)
            self.prob.hessian(self.x, self.hess)

            # Compute the complete KKT residual
            self.optimizer.compute_residual(barrier_param, self.vars, self.grad, self.res)

            # Compute the reduced residual for the right-hand-side of the KKT system
            self.optimizer.compute_reduced_residual(self.vars, self.res, self.bx)

            # Add the diagonal contributions to the Hessian matrix
            self.optimizer.add_diagonal(self.vars, self.hess)

            # Solve the KKT system
            self._solve(self.hess, self.bx, self.px)

            # Compute the full update based on the reduced variable update
            self.optimizer.compute_update_from_reduced(self.vars, self.res, self.px, self.update)

            # Compute the max step in the multipliers 
            alpha_x, alpha_y = self.optimizer.compute_max_step(tau, self.vars, self.update)

            # Compute the full update
            self.optimizer.apply_step_update(alpha_x, alpha_y, self.update, self.vars)


            barrier_param = 0.5 * barrier_param



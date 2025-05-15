import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from mdgo import mdgo


class CartProblem:
    def __init__(self):
        self.N = 201
        self.tf = 2.0
        self.cart = mdgo.CartPoleProblem(self.N, self.tf)

        self.ndof = self.cart.get_num_dof()

        self.mat_obj = self.cart.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = self.mat_obj.get_nonzero_structure()

    def get_init_point(self):
        x = np.zeros(self.ndof)

        # Set the initial guess

        
    

    def gradient(self, x):
        g = self.cart.gradient(x)

        return g
    
    def hessian(self, x):
        self.cart.hessian(x, self.mat_obj)

        data = self.mat_obj.get_data()
        jac = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

        return jac
    
    def optimize(self):


problem = CartProblem()

p = -spsolve(jac, g)
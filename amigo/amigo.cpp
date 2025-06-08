#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cart_pole_problem.h"
#include "csr_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(mdgo, mod) {
  mod.doc() = "MDGO for MDO on GPUs";

  py::class_<mdgo::CSRMat<double>, std::shared_ptr<mdgo::CSRMat<double>>>(
      mod, "CSRMat")
      .def("get_nonzero_structure",
           [](mdgo::CSRMat<double> &mat) {
             py::array_t<int> rowp(mat.nrows + 1);
             std::memcpy(rowp.mutable_data(), mat.rowp,
                         (mat.nrows + 1) * sizeof(int));

             py::array_t<int> cols(mat.nnz);
             std::memcpy(cols.mutable_data(), mat.cols, mat.nnz * sizeof(int));
             return py::make_tuple(mat.nrows, mat.ncols, mat.nnz, rowp, cols);
           })
      .def("get_data", [](mdgo::CSRMat<double> &mat) -> py::array_t<double> {
        py::array_t<double> data(mat.nnz);
        std::memcpy(data.mutable_data(), mat.data, mat.nnz * sizeof(double));
        return data;
      });

  py::class_<mdgo::CartPoleProblem<double>>(mod, "CartPoleProblem")
      .def(py::init<int, double>(), py::arg("N"), py::arg("tf"))
      .def("get_num_dof", &mdgo::CartPoleProblem<double>::get_num_dof)
      .def("lagrangian",
           [](mdgo::CartPoleProblem<double> &self, py::array_t<double> x) {
             int size = self.get_num_dof();
             auto x_vec = self.create_vector();
             std::memcpy(x_vec->get_host_array(), x.data(),
                         size * sizeof(double));
             return self.lagrangian(x_vec);
           })
      .def("gradient",
           [](mdgo::CartPoleProblem<double> &self,
              py::array_t<double> x) -> py::array_t<double> {
             int size = self.get_num_dof();
             auto g_vec = self.create_vector();
             auto x_vec = self.create_vector();

             std::memcpy(x_vec->get_host_array(), x.data(),
                         size * sizeof(double));

             self.gradient(x_vec, g_vec);

             py::array_t<double> g(size);
             std::memcpy(g.mutable_data(), g_vec->get_host_array(),
                         size * sizeof(double));
             return g;
           })
      .def("create_csr_matrix",
           &mdgo::CartPoleProblem<double>::create_csr_matrix)
      .def("hessian", [](mdgo::CartPoleProblem<double> &self,
                         py::array_t<double> x,
                         std::shared_ptr<mdgo::CSRMat<double>> &mat) {
        int size = self.get_num_dof();
        auto x_vec = self.create_vector();
        std::memcpy(x_vec->get_host_array(), x.data(), size * sizeof(double));

        self.hessian(x_vec, mat);
      });
}

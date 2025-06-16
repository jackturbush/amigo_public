#ifndef AMIGO_OPTIMIZATION_PROBLEM_H
#define AMIGO_OPTIMIZATION_PROBLEM_H

#include "component_group_base.h"

namespace amigo {

template <typename T>
class OptimizationProblem {
 public:
  using Vec = std::shared_ptr<Vector<T>>;
  using Mat = std::shared_ptr<CSRMat<T>>;

  OptimizationProblem(int data_size, int num_variables,
                      std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps)
      : data_size(data_size), num_variables(num_variables), comps(comps) {
    data_vec = std::make_shared<Vector<T>>(data_size);
  }

  int get_num_variables() const { return num_variables; }
  Vec create_vector() const {
    return std::make_shared<Vector<T>>(num_variables);
  }
  Vec get_data_vector() { return data_vec; }

  T lagrangian(Vec& x) const {
    T lagrange = 0.0;
    for (size_t i = 0; i < comps.size(); i++) {
      lagrange += comps[i]->lagrangian(*data_vec, *x);
    }
    return lagrange;
  }

  void gradient(const Vec& x, Vec& g) const {
    g->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_gradient(*data_vec, *x, *g);
    }
  }

  void hessian_product(const Vec& x, const Vec& p, Vec& h) const {
    h->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian_product(*data_vec, *x, *p, *h);
    }
  }

  void hessian(const Vec& x, Mat& mat) const {
    mat->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian(*data_vec, *x, *mat);
    }
  }

  std::shared_ptr<CSRMat<T>> create_csr_matrix() const {
    std::set<std::pair<int, int>> node_set;
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_nonzero_pattern(node_set);
    }

    std::vector<int> rowp(num_variables + 1);
    for (auto it = node_set.begin(); it != node_set.end(); it++) {
      rowp[it->first + 1] += 1;
    }

    // Set the pointer into the rows
    rowp[0] = 0;
    for (int i = 0; i < num_variables; i++) {
      rowp[i + 1] += rowp[i];
    }

    int nnz = rowp[num_variables];
    std::vector<int> cols(nnz);

    for (auto it = node_set.begin(); it != node_set.end(); it++) {
      cols[rowp[it->first]] = it->second;
      rowp[it->first]++;
    }

    // Reset the pointer into the nodes
    for (int i = num_variables; i > 0; i--) {
      rowp[i] = rowp[i - 1];
    }
    rowp[0] = 0;

    return std::make_shared<CSRMat<T>>(num_variables, num_variables, nnz,
                                       rowp.data(), cols.data());
  }

 private:
  int data_size;
  int num_variables;
  std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps;
  Vec data_vec;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZATION_PROBLEM_H
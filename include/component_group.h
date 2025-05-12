#ifndef MDGO_COMPONENT_COLLECTION_H
#define MDGO_COMPONENT_COLLECTION_H

#include "vector.h"

namespace mdgo {

template <typename T, class Component, class Layout>
class ParallelComponentCollection {
 public:
  ParallelComponentCollection(Component &comp, Layout &layout) {}

  T lagrangian(Vector<T> &vec) const {
    T value = 0.0;
    for (int i = 0; i < layout.length(); i++) {
      Component::Input<T> input;
      layout.get_values(i, vec, input);
      value += comp.lagrange(input);
    }
  }

  void add_gradient(Vector<T> &vec, Vector<T> &res) const {
    Component::Input<T> input, gradient;
    for (int i = 0; i < layout.length(); i++) {
      gradient.zero();
      layout.get_values(i, vec, input);
      comp.gradient(input, gradient);
      layout.add_values(i, gradient, res);
    }
  }
};

}  // namespace mdgo

#endif  // MDGO_COMPONENT_COLLECTION_H
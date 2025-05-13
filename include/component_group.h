#ifndef MDGO_COMPONENT_COLLECTION_H
#define MDGO_COMPONENT_COLLECTION_H

#include "vector.h"

namespace mdgo {

template <typename T, class Component, class Layout>
class SerialCollection {
 public:
  using Input = typename Component::template Input<T>;

  SerialCollection(Component &comp, Layout &layout)
      : comp(comp), layout(layout) {}

  T lagrangian(const Vector<T> &vec) const {
    T value = 0.0;
    for (int i = 0; i < layout.get_length(); i++) {
      Input input;
      layout.get_values(i, vec, input);
      value += comp.lagrange(input);
    }
    return value;
  }

  void add_gradient(const Vector<T> &vec, Vector<T> &res) const {
    Input input, gradient;
    for (int i = 0; i < layout.get_length(); i++) {
      gradient.zero();
      layout.get_values(i, vec, input);
      comp.gradient(input, gradient);
      layout.add_values(i, gradient, res);
    }
  }

  // void hessian_product()

 private:
  Component &comp;
  Layout &layout;
};

}  // namespace mdgo

#endif  // MDGO_COMPONENT_COLLECTION_H
#include "cart_component.h"
#include "component_group.h"
#include "layout.h"
#include "vector.h"

int main(int argc, char *argv[]) {
  using T = double;

  using Component = mdgo::CartPoleComponent<T>;
  using Input = Component::template Input<T>;
  using Layout = mdgo::IndexLayout<Component>;

  T g = 9.81;
  T L = 0.5;
  T m1 = 1.0;
  T m2 = 0.5;

  mdgo::CartPoleComponent<T> cart(g, L, m1, m2);

  int N = 201;  // Number of time levels
  mdgo::Vector<int, Component::ncomp> indices(N);

  indices.host_initialize();
  int *array = indices.get_host_array();
  for (int i = 0; i < indices.get_size(); i++) {
    array[i] = i;
  }

  int ndof = indices.get_size();

  mdgo::IndexLayout<mdgo::CartPoleComponent<T>> layout(indices);

  mdgo::SerialCollection<T, Component, Layout> collect(cart, layout);

  mdgo::Vector<T> x(ndof);
  mdgo::Vector<T> grad(ndof);
  mdgo::Vector<T> p(ndof);

  x.host_initialize();
  grad.host_initialize();
  p.host_initialize();

  x.set_random();
  p.set_random();
  grad.zero();

  T L1 = collect.lagrangian(x);
  collect.add_gradient(x, grad);

  double dh = 1e-6;
  x.axpy(dh, p);
  T L2 = collect.lagrangian(x);

  T fd = (L2 - L1) / dh;
  T ans = grad.dot(p);

  std::printf("%12.4e   %12.4e    %12.4e\n", ans, fd, (ans - fd) / fd);

  // double dh = 1e-6;
  // for (int i = 0; i < Input::ncomp; i++) {
  //   T tmp = input[i];
  //   input[i] = tmp + dh;
  //   T L2 = cart.lagrange(input);
  //   input[i] = tmp;

  //   T fd = (L2 - L1) / dh;

  //   std::printf("%12.4e   %12.4e    %12.4e\n", grad[i], fd,
  //               (grad[i] - fd) / fd);
  // }

  return 0;
}
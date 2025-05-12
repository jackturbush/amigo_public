#include "cart_component.h"
// #include "layout.h"
// #include "vector.h"

int main(int argc, char *argv[]) {
  using T = double;

  using Input = mdgo::CartPoleComponent<T>::Input<T>;

  T g = 9.81;
  T L = 0.5;
  T m1 = 1.0;
  T m2 = 0.5;

  mdgo::CartPoleComponent<T> cart(g, L, m1, m2);

  Input input, grad;

  input.set_rand();

  T L1 = cart.lagrange(input);
  cart.gradient(input, grad);

  double dh = 1e-6;
  for (int i = 0; i < Input::ncomp; i++) {
    T tmp = input[i];
    input[i] = tmp + dh;
    T L2 = cart.lagrange(input);
    input[i] = tmp;

    T fd = (L2 - L1) / dh;

    std::printf("%12.4e   %12.4e    %12.4e\n", grad[i], fd,
                (grad[i] - fd) / fd);
  }

  return 0;
}
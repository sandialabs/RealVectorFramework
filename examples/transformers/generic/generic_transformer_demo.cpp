#include "generic_transformer.hpp"
#include <iostream>

int main() {
  using Vector = std::vector<double>;
  using Matrix = transformer::Matrix<Vector>;

  constexpr std::size_t d_model = 64;

  // Build a generic transformer for Vector
  auto block = transformer::make_transformer_block<Vector>(d_model, 4, 256);
  auto model = transformer::make_transformer<Vector>(d_model, 2, 4, 256);

  Vector x(d_model, 1.0);
  Vector y(d_model, 0.0);

  block(y, x);
  std::cout << "Block output norm^2: " << rvf::inner_product(y, y) << "\n";

  Vector z(d_model, 0.0);
  model(z, x);
  std::cout << "Model output norm^2: " << rvf::inner_product(z, z) << "\n";

  static_assert(rvf::self_map_c<decltype(block), Vector>);
  static_assert(rvf::self_map_c<decltype(model), Vector>);

  return 0;
}


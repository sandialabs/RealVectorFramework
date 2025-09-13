#include "sequence_transformer.hpp"
#include <iostream>

int main() {
  using Vector = std::vector<double>;
  using Sequence = seqtrf::Sequence<Vector>;

  const std::size_t d_model = 32;
  const std::size_t L = 4; // sequence length

  // Build a small sequence transformer (2 layers)
  auto block = seqtrf::make_sequence_block<Vector>(d_model, 4 * d_model);
  auto model = seqtrf::make_sequence_transformer<Vector>(d_model, 2, 4 * d_model);

  // Make an input sequence of constant tokens
  Sequence x(L, Vector(d_model, 1.0));
  Sequence y = x;

  block(y, x);
  double block_energy = 0.0;
  for (const auto& tok : y) block_energy += rvf::inner_product(tok, tok);
  std::cout << "Block output energy: " << block_energy << "\n";

  Sequence z = x;
  model(z, x);
  double model_energy = 0.0;
  for (const auto& tok : z) model_energy += rvf::inner_product(tok, tok);
  std::cout << "Model output energy: " << model_energy << "\n";

  static_assert(rvf::self_map_c<decltype(block), Sequence>);
  static_assert(rvf::self_map_c<decltype(model), Sequence>);
  return 0;
}


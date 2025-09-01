/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <iostream>
#include <vector>
#include <numeric>
#include "rvf.hpp"

// The generic axpy algorithm (y = alpha*x + y)
template<typename V, typename S>
  requires rvf::real_vector_c<V> && real_scalar_c<S>
void axpy(S alpha, const V& x, V& y) {
  // Create a temporary copy of x to scale, leaving the original x unmodified.
  auto temp = rvf::clone(x);
  rvf::scale_in_place(temp, alpha);
  rvf::add_in_place(y, temp);
}

int main() {
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {4.0, 5.0, 6.0};
  double alpha = 2.0;

  std::cout << "Before axpy:" << std::endl;
  std::cout << "x: ";
  for (const auto& val : x) { std::cout << val << " "; }
  std::cout << std::endl;
  std::cout << "y: ";
  for (const auto& val : y) { std::cout << val << " "; }
  std::cout << std::endl;

  axpy(alpha, x, y);

  std::cout << "\nAfter axpy:" << std::endl;
  std::cout << "y: ";
  for (const auto& val : y) { std::cout << val << " "; }
  std::cout << std::endl;

  return 0;
}

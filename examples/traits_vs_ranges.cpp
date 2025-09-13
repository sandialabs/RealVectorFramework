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

#include "core/rvf.hpp"

// Include the illustrative trait + shim implementations for std::vector.
// This routes RVF CPO calls on std::vector through tincup::cpo_impl
// specializations defined in this header, instead of the generic ranges path.
#include "core/type_support/std_cpo_impl.hpp"

int main() {
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {4.0, 5.0, 6.0};
  double alpha = 2.0;

  std::cout << "Example (traits path) before axpy_in_place:" << std::endl;
  std::cout << "x: "; for (const auto& v : x) std::cout << v << ' '; std::cout << '\n';
  std::cout << "y: "; for (const auto& v : y) std::cout << v << ' '; std::cout << '\n';

  // Use the in-place AXPY CPO (implemented using other CPOs)
  const auto& x_const = x;  // Ensure const reference
  rvf::axpy_in_place(y, alpha, x_const);

  std::cout << "After axpy_in_place (traits path):\n";
  std::cout << "y: "; for (const auto& v : y) std::cout << v << ' '; std::cout << '\n';

  // Quick sanity: y should be 2*x + original y => {6,9,12}
  if (y != std::vector<double>({6.0, 9.0, 12.0})) return 1;
  return 0;
}


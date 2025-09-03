/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include "conjugate_gradient.hpp"
#include "real_vector.hpp"

#include <vector>
#include <numeric>
#include <iostream>
#include <memory>

// A simple wrapper around std::vector to make it a real_vector_c
struct StdVectorWrapper {
  std::vector<double> data;
};

// tag_invoke implementations for StdVectorWrapper
namespace rvf {

// add_in_place
void tag_invoke(add_in_place_ftor, StdVectorWrapper& target, const StdVectorWrapper& source) {
  for (size_t i = 0; i < target.data.size(); ++i) {
    target.data[i] += source.data[i];
  }
}



// clone
StdVectorWrapper tag_invoke(clone_ftor, const StdVectorWrapper& source) {
  return {source.data};
}

// dimension
size_t tag_invoke(dimension_ftor, const StdVectorWrapper& vec) {
  return vec.data.size();
}

// inner_product
double tag_invoke(inner_product_ftor, const StdVectorWrapper& lhs, const StdVectorWrapper& rhs) {
  double result = 0.0;
  for (size_t i = 0; i < lhs.data.size(); ++i) {
    result += lhs.data[i] * rhs.data[i];
  }
  return result;
}

// scale_in_place
void tag_invoke(scale_in_place_ftor, StdVectorWrapper& target, double scalar) {
  for (auto& val : target.data) {
    val *= scalar;
  }
}

} // namespace rvf

// A simple matrix for testing: a diagonal matrix.
struct DiagonalMatrix {
  std::vector<double> diag;

  void operator()(StdVectorWrapper& y, const StdVectorWrapper& x) const {
    for (size_t i = 0; i < x.data.size(); ++i) {
      y.data[i] = diag[i] * x.data[i];
    }
  }
};

void print_vector(const StdVectorWrapper& vec) {
  std::cout << "[ ";
  for (const auto& val : vec.data) {
    std::cout << val << " ";
  }
  std::cout << "]" << std::endl;
}

int main() {
  DiagonalMatrix A;
  A.diag = {4.0, 4.0, 4.0};

  StdVectorWrapper b;
  b.data = {1.0, 2.0, 3.0};

  StdVectorWrapper x;
  x.data = {0.0, 0.0, 0.0};

  std::cout << "Solving Ax = b for:" << std::endl;
  std::cout << "A = diagonal matrix with " << A.diag[0] << " on diagonal" << std::endl;
  std::cout << "b = "; print_vector(b);
  std::cout << "Initial x = "; print_vector(x);

  rvf::conjugate_gradient(A, b, x);

  std::cout << "Solution x = "; print_vector(x);
  
  // Expected solution for this diagonal system is x_i = b_i / A_ii
  std::cout << "Expected x = [ 0.25 0.5 0.75 ]" << std::endl;

  return 0;
}

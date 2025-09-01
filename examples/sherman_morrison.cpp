/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include "sherman_morrison.hpp"
#include "real_vector.hpp"

#include <vector>
#include <iostream>
#include <memory>
#include <random>

// Reuse the StdVectorWrapper from conjugate_gradient.cpp
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

// clone - returns by value (not pointer)
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

// A simple diagonal matrix inverse operator (for general Sherman-Morrison test)
struct DiagonalInverseOperator {
  std::vector<double> inv_diag;

  void operator()(StdVectorWrapper& y, const StdVectorWrapper& x) const {
    for (size_t i = 0; i < x.data.size(); ++i) {
      y.data[i] = inv_diag[i] * x.data[i];
    }
  }
};

void print_vector(const StdVectorWrapper& vec, const std::string& name = "") {
  if (!name.empty()) {
    std::cout << name << " = ";
  }
  std::cout << "[ ";
  for (const auto& val : vec.data) {
    std::cout << val << " ";
  }
  std::cout << "]" << std::endl;
}

// Test wrapper that handles both value and pointer returns from clone
template<typename Vec>
auto safe_clone_and_deref(const Vec& v) {
  auto clone_deref = tincup::auto_deref_result(rvf::clone);
  return clone_deref(v);
}

void test_identity_plus_rank1() {
  std::cout << "\n=== Testing Sherman-Morrison for (I + u*v^T)x = b ===" << std::endl;
  
  // Create test vectors
  StdVectorWrapper u, v, b, x;
  u.data = {1.0, 0.5, 0.0};
  v.data = {0.0, 1.0, 2.0};  
  b.data = {3.0, 2.0, 1.0};
  x.data = {0.0, 0.0, 0.0};  // Initialize solution vector
  
  print_vector(u, "u");
  print_vector(v, "v");
  print_vector(b, "b");
  
  // Solve (I + u*v^T)x = b
  rvf::sherman_morrison_identity_plus_rank1(u, v, b, x);
  
  print_vector(x, "Solution x");
  
  // Verify the solution by computing (I + u*v^T)*x
  StdVectorWrapper verification;
  verification.data = {0.0, 0.0, 0.0};
  
  // First compute I*x = x  
  verification = x;
  
  // Then add u*(v^T*x)
  double vtx = rvf::inner_product(v, x);
  auto u_scaled = safe_clone_and_deref(u);
  rvf::scale_in_place(u_scaled, vtx);
  rvf::add_in_place(verification, u_scaled);
  
  print_vector(verification, "Verification: (I + u*v^T)*x");
  std::cout << "Should match b above.\n" << std::endl;
}

void test_general_sherman_morrison() {
  std::cout << "=== Testing General Sherman-Morrison for (A + u*v^T)x = b ===" << std::endl;
  
  // Use a diagonal matrix A with diagonal [2, 3, 4]
  DiagonalInverseOperator A_inv;
  A_inv.inv_diag = {0.5, 1.0/3.0, 0.25};  // A^(-1) diagonal
  
  StdVectorWrapper u, v, b, x;
  u.data = {1.0, 1.0, 1.0};
  v.data = {0.5, 0.5, 0.5};
  b.data = {6.0, 9.0, 8.0};
  x.data = {0.0, 0.0, 0.0};
  
  print_vector(u, "u");
  print_vector(v, "v"); 
  print_vector(b, "b");
  std::cout << "A^(-1) diagonal = [ 0.5 0.333 0.25 ]" << std::endl;
  
  rvf::sherman_morrison_general(A_inv, u, v, b, x);
  
  print_vector(x, "Solution x");
  
  // Basic verification: compute A*x (without the rank-1 update)
  StdVectorWrapper Ax;
  Ax.data = {x.data[0] * 2.0, x.data[1] * 3.0, x.data[2] * 4.0};
  print_vector(Ax, "A*x (without rank-1 update)");
  std::cout << std::endl;
}

void test_multiple_rhs() {
  std::cout << "=== Testing Multiple RHS Sherman-Morrison ===" << std::endl;
  
  StdVectorWrapper u, v;
  u.data = {1.0, 0.0, 0.0};
  v.data = {0.0, 1.0, 0.0};
  
  // Multiple right-hand sides
  std::vector<StdVectorWrapper> B(2), X(2);
  B[0].data = {1.0, 2.0, 3.0};
  B[1].data = {4.0, 5.0, 6.0};
  
  X[0].data = {0.0, 0.0, 0.0};
  X[1].data = {0.0, 0.0, 0.0};
  
  print_vector(u, "u");
  print_vector(v, "v");
  print_vector(B[0], "b1");
  print_vector(B[1], "b2");
  
  rvf::sherman_morrison_multiple_rhs(u, v, B, X);
  
  print_vector(X[0], "Solution x1");
  print_vector(X[1], "Solution x2");
  std::cout << std::endl;
}

int main() {
  std::cout << "Sherman-Morrison Formula Solver Examples" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  try {
    test_identity_plus_rank1();
    test_general_sherman_morrison();
    test_multiple_rhs();
    
    std::cout << "All tests completed successfully!" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

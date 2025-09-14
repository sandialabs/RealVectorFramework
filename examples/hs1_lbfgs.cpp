/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include "core/rvf.hpp"
#include "algorithms/optimization/lbfgs.hpp"

// Hock-Schittkowski Problem #1: min 100*(x2 - x1^2)^2 + (1 - x1)^2
// This is the unconstrained version of the problem.
class HS1Problem {
public:
  using value_type = double;
  
  template<rvf::real_vector_c Vec>
  value_type value(const Vec& x) const {
    // f(x) = 100*(x[1] - x[0]^2)^2 + (1 - x[0])^2
    auto x0 = x[0];
    auto x1 = x[1];
    auto term1 = x1 - x0 * x0;
    auto term2 = 1.0 - x0;
    return 100.0 * term1 * term1 + term2 * term2;
  }
  
  template<rvf::real_vector_c Vec>
  void gradient(Vec& grad, const Vec& x) const {
    // ∇f = [-400*(x[1] - x[0]^2)*x[0] - 2*(1 - x[0]), 200*(x[1] - x[0]^2)]
    auto x0 = x[0];
    auto x1 = x[1];
    auto term = x1 - x0 * x0;
    
    grad[0] = -400.0 * term * x0 - 2.0 * (1.0 - x0);
    grad[1] = 200.0 * term;
  }
};

int main() {
  using Vec = std::vector<double>;
  
  // Problem setup
  HS1Problem problem;
  
  // Initial point: x0 = [-2.0, 1.0]
  Vec x = {-2.0, 1.0};
  
  std::cout << "Hock-Schittkowski Problem #1 (Unconstrained)" << std::endl;
  std::cout << "min f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2" << std::endl;
  std::cout << "Known optimal solution: x* = [1.0, 1.0], f* = 0" << std::endl << std::endl;
  
  std::cout << "Initial point: x0 = [" << x[0] << ", " << x[1] << "]" << std::endl;
  std::cout << "Initial objective: f(x0) = " << problem.value(x) << std::endl << std::endl;
  
  // Solve using L-BFGS
  rvf::lbfgs(
    problem,
    x,
    5,      // history size
    1e-6,   // gradient tolerance
    1000    // max iterations
  );
  
  std::cout << "Final solution: x* = [" << x[0] << ", " << x[1] << "]" << std::endl;
  std::cout << "Final objective: f(x*) = " << problem.value(x) << std::endl;
  
  // Compute final gradient to check optimality
  Vec grad = {0.0, 0.0};
  problem.gradient(grad, x);
  auto grad_norm = rvf::sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
  std::cout << "Final gradient norm: ||∇f(x*)|| = " << grad_norm << std::endl;
  
  // Check how close we are to known optimal solution [1.0, 1.0]
  auto error_x0 = std::abs(x[0] - 1.0);
  auto error_x1 = std::abs(x[1] - 1.0);
  std::cout << "Error from known optimum: |x0 - 1| = " << error_x0 
        << ", |x1 - 1| = " << error_x1 << std::endl;
  
  return 0;
}

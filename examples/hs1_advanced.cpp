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
#include <iomanip>
#include "core/rvf.hpp"
#include "algorithms/optimization/line_search/gradient_descent_bounds.hpp"

// HS1 Problem with iteration tracking
class HS1Problem {
public:
  using value_type = double;
  mutable int nfeval = 0;
  mutable int ngeval = 0;
  
  template<rvf::real_vector_c Vec>
  value_type value(const Vec& x) const {
    ++nfeval;
    auto x0 = x[0];
    auto x1 = x[1];
    auto term1 = x1 - x0 * x0;
    auto term2 = 1.0 - x0;
    return 100.0 * term1 * term1 + term2 * term2;
  }
  
  template<rvf::real_vector_c Vec>
  void gradient(Vec& grad, const Vec& x) const {
    ++ngeval;
    auto x0 = x[0];
    auto x1 = x[1];
    auto term = x1 - x0 * x0;
    
    grad[0] = -400.0 * term * x0 - 2.0 * (1.0 - x0);
    grad[1] = 200.0 * term;
  }
  
  void reset_counters() {
    nfeval = ngeval = 0;
  }
};

// Custom gradient descent with iteration callback
template<typename Objective, rvf::real_vector_c Vec, typename Callback>
requires rvf::objective_function_c<Objective, Vec>
void gradient_descent_bounds_verbose(
  const Objective& obj,
  const rvf::bound_constraints<Vec>& bounds,
  Vec& x,
  Callback callback,
  rvf::vector_value_t<Vec> gtol = 1e-6,
  rvf::vector_value_t<Vec> alpha0 = 1.0,
  rvf::vector_size_t<Vec> maxIter = 1000) {

  using value_type = rvf::vector_value_t<Vec>;
  
  // Initialize linesearch parameters  
  rvf::linesearch_params<value_type> ls_params;
  ls_params.alpha_init = alpha0;
  ls_params.c1 = 1e-4;    // More aggressive sufficient decrease
  ls_params.rho = 0.5;    // Standard backtracking factor
  
  bounds.project(x);
  
  auto grad = rvf::clone(x);
  auto x_trial = rvf::clone(x);
  
  for (rvf::vector_size_t<Vec> iter = 0; iter < maxIter; ++iter) {
  auto f_x = obj.value(x);
  obj.gradient(grad, x);
  
  auto grad_dot_grad = rvf::inner_product(grad, grad);
  auto grad_norm = rvf::sqrt(grad_dot_grad);
  
  // Call iteration callback
  callback(iter, f_x, grad_norm, x);
  
  if (grad_norm < gtol) {
    break;
  }
  
  auto alpha = rvf::backtracking_linesearch(
    obj, bounds, x, grad, x_trial, f_x, grad_dot_grad, ls_params);
  
  x = std::move(x_trial);
  
  if (ls_params.adaptive_init) {
    ls_params.alpha_init = alpha;
  }
  }
}

int main() {
  using Vec = std::vector<double>;
  
  HS1Problem problem;
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Enhanced HS1 Gradient Descent with ROL-style Linesearch\n";
  std::cout << "======================================================\n\n";
  
  // Test different starting points
  std::vector<Vec> starting_points = {
    {-2.0, 1.0},  // Original HS1 starting point
    {0.0, 0.0},   // Origin
    {2.0, 3.0},   // Different quadrant
    {-1.0, -1.0}  // Constrained starting point
  };
  
  Vec lower = {-std::numeric_limits<double>::infinity(), -1.5};
  Vec upper = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  rvf::bound_constraints<Vec> bounds{lower, upper};
  
  for (size_t test = 0; test < starting_points.size(); ++test) {
    Vec x = starting_points[test];
    problem.reset_counters();
    
    std::cout << "Test " << (test+1) << ": Starting point [" 
          << x[0] << ", " << x[1] << "]\n";
    std::cout << "Initial objective: " << problem.value(x) << "\n";
    
    // Iteration callback for verbose output
    auto callback = [](int iter, double f, double gnorm, const Vec& x) {
      if (iter % 10 == 0 || iter < 5) {
        std::cout << "  Iter " << std::setw(3) << iter 
             << ": f = " << std::setw(12) << f
             << ", ||∇f|| = " << std::setw(10) << gnorm
             << ", x = [" << std::setw(8) << x[0] 
             << ", " << std::setw(8) << x[1] << "]\n";
      }
    };
    
    gradient_descent_bounds_verbose(
      problem, bounds, x, callback,
      1e-8,  // Tighter tolerance
      1.0,   // Initial step size
      500    // Max iterations
    );
    
    Vec grad = {0.0, 0.0};
    problem.gradient(grad, x);
    double grad_norm = rvf::sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
    
    std::cout << "Final: f = " << problem.value(x) 
         << ", ||∇f|| = " << grad_norm << "\n";
    std::cout << "Solution: [" << x[0] << ", " << x[1] << "]\n";
    std::cout << "Function evals: " << problem.nfeval 
         << ", Gradient evals: " << problem.ngeval << "\n";
    std::cout << "Error from [1,1]: [" << std::abs(x[0]-1.0) 
         << ", " << std::abs(x[1]-1.0) << "]\n\n";
  }
  
  return 0;
}

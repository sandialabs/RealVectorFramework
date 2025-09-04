/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "operations/advanced/binary_in_place.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include <cmath>
#include <algorithm>

namespace rvf {

// Objective function concept
template<typename F, typename Vec>
concept objective_function_c = requires(const F& f, const Vec& x, Vec& grad) {
  { f.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
  { f.gradient(grad, x) } -> std::same_as<void>; // computes gradient into grad
};

// Bound constraints representation
template<real_vector_c Vec>
struct bound_constraints {
  Vec lower, upper;
  
  // Project x onto [lower, upper] bounds
  void project(Vec& x) const {
  binary_in_place(x, lower, [](auto xi, auto li) { return std::max(xi, li); });
  binary_in_place(x, upper, [](auto xi, auto ui) { return std::min(xi, ui); });
  }
  
  // Check if all bounds are satisfied
  bool is_feasible(const Vec& x) const {
  // Could implement using reduction, but keeping it simple
  return true; // Simplified for demonstration
  }
};

// Linesearch parameters structure (ROL-style)
template<typename T>
struct linesearch_params {
  T c1 = T(1e-4);       // Armijo sufficient decrease parameter
  T rho = T(0.5);       // Backtracking reduction factor  
  T alpha_init = T(1.0);  // Initial step size
  T alpha_min = T(1e-12);   // Minimum step size
  int max_eval = 20;    // Maximum function evaluations
  bool adaptive_init = true; // Use previous successful step for initialization
};

// ROL-style backtracking linesearch for bound-constrained problems
template<typename Objective, real_vector_c Vec>
requires objective_function_c<Objective, Vec>
vector_value_t<Vec> backtracking_linesearch(
  const Objective& obj,
  const bound_constraints<Vec>& bounds,
  const Vec& x,
  const Vec& grad,
  Vec& x_trial,
  vector_value_t<Vec> f_x,
  vector_value_t<Vec> grad_dot_grad,
  const linesearch_params<vector_value_t<Vec>>& ls_params) {
  
  using value_type = vector_value_t<Vec>;
  
  value_type alpha = ls_params.alpha_init;
  int neval = 0;
  
  for (int iter = 0; iter < ls_params.max_eval; ++iter) {
  // Projected gradient step: x_trial_local = P(x - alpha * grad)
  auto x_cl = clone(x); auto& x_trial_local = deref_if_needed(x_cl);

  binary_in_place(x_trial_local, grad, [alpha](auto xi, auto gi) { 
    return xi - alpha * gi; 
  });
  bounds.project(x_trial_local);
  
  // Evaluate objective at trial point
  auto f_trial = obj.value(x_trial_local);
  ++neval;
  
  // Armijo sufficient decrease condition
  value_type armijo_rhs = f_x - ls_params.c1 * alpha * grad_dot_grad;
  
  if (f_trial <= armijo_rhs || alpha <= ls_params.alpha_min) {
    x_trial = std::move(x_trial_local); // write back to output parameter
    return alpha; // Accept step
  }
  
  // Reduce step size
  alpha *= ls_params.rho;
  }
  
  return ls_params.alpha_min; // Fallback minimal step
}

/**
 * @brief Solves min f(x) s.t. lower <= x <= upper using projected gradient descent
 * with ROL-style backtracking linesearch
 */
template<typename Objective, real_vector_c Vec>
requires objective_function_c<Objective, Vec>
void gradient_descent_bounds(
  const Objective& obj,
  const bound_constraints<Vec>& bounds,
  Vec& x,
  vector_value_t<Vec> gtol = 1e-6,
  vector_value_t<Vec> alpha0 = 1.0,
  vector_size_t<Vec> maxIter = 1000) {

  using value_type = vector_value_t<Vec>;
  
  // Initialize linesearch parameters
  linesearch_params<value_type> ls_params;
  ls_params.alpha_init = alpha0;
  
  // Project initial point to feasible region
  bounds.project(x);
  
  auto x_cl1 = clone(x); auto& grad    = deref_if_needed(x_cl1);
  auto x_cl2 = clone(x); auto& x_trial = deref_if_needed(x_cl2);
  
  for(vector_size_t<Vec> iter = 0; iter < maxIter; ++iter) {
    // Compute objective and gradient at current point
    auto f_x = obj.value(x);
    obj.gradient(grad, x);
    
    // Check convergence
    auto grad_dot_grad = inner_product(grad, grad);
    auto grad_norm = std::sqrt(grad_dot_grad);
    if (grad_norm < gtol) {
      break;
    }
    
    // Perform backtracking linesearch
    auto alpha = backtracking_linesearch(
      obj, bounds, x, grad, x_trial, f_x, grad_dot_grad, ls_params);
    
    // Update iterate (x_trial was computed in linesearch)
    x = std::move(x_trial);
    
    // Adaptive step size initialization (ROL-style)
    if (ls_params.adaptive_init) {
      ls_params.alpha_init = alpha;
    }
  }
} // gradient_descent_bounds

} // namespace rvf

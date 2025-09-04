/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "operations/advanced/axpy_in_place.hpp"
#include "objectives/objective.hpp"
#include <cmath>
#include <algorithm>

namespace rvf {

template<real_vector_c Vec>
using vector_value_t = inner_product_return_t<Vec>;

template<real_vector_c Vec>
using vector_size_t = dimension_return_t<Vec>;

/**
 * @brief Identity preconditioner (no-op)
 */
template<real_vector_c Vec>
struct identity_preconditioner {
  void operator()(Vec& y, const Vec& x) const {
    y = x;  // Assumes assignment operator exists
  }
};

/**
 * @brief Steihaug-Toint Truncated Conjugate Gradient algorithm for trust region subproblems
 * 
 * Solves the trust region subproblem:
 *   min_s  g^T s + (1/2) s^T H s
 *   s.t.   ||s|| <= delta
 * 
 * Based on the ROL implementation but adapted to RealVectorFramework patterns.
 */
template<typename Obj, real_vector_c Vec, typename Precond = identity_preconditioner<Vec>>
requires objective_value_c<Obj, Vec> && objective_gradient_c<Obj, Vec> && objective_hess_vec_c<Obj, Vec> && self_map_c<Precond, Vec>
class TruncatedCG {
public:
  using value_type = vector_value_t<Vec>;
  using size_type = vector_size_t<Vec>;
  
  enum class TerminationStatus {
    CONVERGED = 0,        // Converged to tolerance
    MAX_ITER = 1,         // Maximum iterations reached
    NEGATIVE_CURVATURE = 2, // Negative curvature detected
    TRUST_REGION_BOUNDARY = 3, // Trust region boundary hit
    PRECONDITIONER_FAILURE = 4 // Preconditioner failure
  };
  
  struct Params {
    value_type abs_tol = value_type(1e-4);        // Absolute tolerance for residual norm
    value_type rel_tol = value_type(1e-2);        // Relative tolerance for residual norm  
    size_type max_iter = 20;                      // Maximum CG iterations
    bool use_preconditioner = false;              // Whether to use preconditioning
  };
  
  struct Result {
    value_type step_norm = value_type(0);         // Norm of computed step
    value_type pred_reduction = value_type(0);    // Predicted reduction in model
    TerminationStatus status = TerminationStatus::CONVERGED; // Termination status
    size_type iter = 0;                           // Number of iterations performed
  };

private:
  const Obj& objective_;
  clone_return_t<Vec> r_;    // residual
  clone_return_t<Vec> v_;    // preconditioned residual
  clone_return_t<Vec> p_;    // search direction
  clone_return_t<Vec> Hp_;   // H*p
  clone_return_t<Vec> s_tmp_; // temporary step
  clone_return_t<Vec> grad_; // gradient storage
  
public:
  /**
   * @brief Constructor that stores reference to objective and allocates working vectors
   * @param obj Objective function satisfying the required concepts
   * @param vec_template Template vector used for cloning working vectors
   */
  TruncatedCG(const Obj& obj, const Vec& vec_template) 
    : objective_(obj),
      r_(clone(vec_template)),
      v_(clone(vec_template)),
      p_(clone(vec_template)), 
      Hp_(clone(vec_template)),
      s_tmp_(clone(vec_template)),
      grad_(clone(vec_template)) {}
  
  /**
   * @brief Solve trust region subproblem
   * @param x Current iterate
   * @param s Output: computed step (input: initial guess, typically zero)
   * @param delta Trust region radius
   * @param params Algorithm parameters
   * @param precond Preconditioner operator (expected to approximate H^{-1})
   * @return Result information including termination status
   */
  Result solve(const Vec& x, Vec& s, value_type delta, 
               const Params& params = {}, 
               const Precond& precond = identity_preconditioner<Vec>{}) {
    
    const auto& obj = objective_;
    auto& r = deref_if_needed(r_);
    auto& v = deref_if_needed(v_);
    auto& p = deref_if_needed(p_);
    auto& Hp = deref_if_needed(Hp_);
    auto& s_tmp = deref_if_needed(s_tmp_);
    auto& g = deref_if_needed(grad_);
    
    const value_type zero(0), one(1), half(0.5);
    
    Result result;
    
    // Compute gradient at current point
    obj.gradient(g, x);
    
    // Initialize step to zero
    s = g; // Use g as template for s structure
    scale_in_place(s, zero); // s = 0
    
    value_type snorm2 = zero;
    
    // Initialize residual r = g (since s = 0 initially)
    r = g;
    value_type gnorm = std::sqrt(inner_product(r, r));
    value_type normg = gnorm;
    
    // Compute tolerance
    const value_type gtol = std::min(params.abs_tol, params.rel_tol * gnorm);
    
    // Apply preconditioner: v = M^{-1} * r
    if (params.use_preconditioner) {
      precond(v, r);
    } else {
      v = r;
    }
    
    // Initialize search direction p = -v
    p = v;
    scale_in_place(p, -one);
    
    // Check if preconditioned system is positive definite
    value_type rho = inner_product(v, r);  // v^T r
    if (rho <= zero) {
      result.status = TerminationStatus::PRECONDITIONER_FAILURE;
      result.iter = 0;
      return result;
    }
    
    // Main CG iteration
    value_type alpha, beta, kappa, sigma, sMp, pnorm2;
    result.pred_reduction = zero;
    
    for (size_type iter = 0; iter < params.max_iter; ++iter) {
      result.iter = iter;
      
      // Compute H*p using Hessian-vector product
      obj.hessVec(Hp, p, x);
      
      // Check for negative curvature
      kappa = inner_product(p, Hp);  // p^T H p
      if (kappa <= zero) {
        // Negative curvature: find boundary of trust region
        sMp = inner_product(s, p);
        pnorm2 = inner_product(p, p);
        sigma = (-sMp + std::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
        axpy_in_place(s, sigma, p);  // s = s + sigma * p
        result.step_norm = delta;
        result.status = TerminationStatus::NEGATIVE_CURVATURE;
        result.pred_reduction += sigma * (rho - half * sigma * kappa);
        return result;
      }
      
      // Compute step length
      alpha = rho / kappa;
      
      // Compute trial step
      s_tmp = s;
      axpy_in_place(s_tmp, alpha, p);  // s_tmp = s + alpha * p
      
      // Check trust region constraint
      value_type s1norm2 = inner_product(s_tmp, s_tmp);
      if (s1norm2 >= delta * delta) {
        // Step would exceed trust region
        sMp = inner_product(s, p);
        pnorm2 = inner_product(p, p);
        sigma = (-sMp + std::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
        axpy_in_place(s, sigma, p);  // s = s + sigma * p
        result.step_norm = delta;
        result.status = TerminationStatus::TRUST_REGION_BOUNDARY;
        result.pred_reduction += sigma * (rho - half * sigma * kappa);
        return result;
      }
      
      // Accept the step
      result.pred_reduction += half * alpha * rho;
      s = s_tmp;
      snorm2 = s1norm2;
      
      // Update residual
      axpy_in_place(r, alpha, Hp);  // r = r + alpha * Hp
      
      // Check convergence
      normg = std::sqrt(inner_product(r, r));
      if (normg < gtol) {
        result.step_norm = std::sqrt(snorm2);
        result.status = TerminationStatus::CONVERGED;
        result.iter = iter + 1;
        return result;
      }
      
      // Apply preconditioner to updated residual
      if (params.use_preconditioner) {
        precond(v, r);
      } else {
        v = r;
      }
      
      // Compute new rho and beta
      value_type rho_new = inner_product(v, r);
      beta = rho_new / rho;
      rho = rho_new;
      
      // Update search direction: p = -v + beta * p
      scale_in_place(p, beta);
      axpy_in_place(p, -one, v);  // p = beta * p - v
    }
    
    // Maximum iterations reached
    result.step_norm = std::sqrt(snorm2);
    result.status = TerminationStatus::MAX_ITER;
    
    return result;
  }
};

} // namespace rvf

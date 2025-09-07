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
#include "operations/advanced/l2norm.hpp"
#include "algorithms/optimization/objective.hpp"
#include "algorithms/optimization/problems.hpp"
#include <cmath>
#include <algorithm>

namespace rvf {

/**
 * @brief Steihaug-Toint Truncated Conjugate Gradient algorithm for trust region subproblems
 * 
 * Solves the trust region subproblem:
 *   min_s  g^T s + (1/2) s^T H s
 *   s.t.   ||s|| <= delta
 * 
 * Based on the ROL implementation but adapted to RealVectorFramework patterns.
 */
template<unconstrained_problem_c Problem>
class TruncatedCG {
public:
	
  using objective_type = objective_t<Problem>;	
  using vector_type    = x0_t<Problem>;
  using value_type     = inner_product_return_t<vector_type>;
  using size_type      = dimension_return_t<vector_type>;
  
  enum class TerminationStatus {
    CONVERGED = 0,        
    MAX_ITER = 1,         
    NEGATIVE_CURVATURE = 2, 
    TRUST_REGION_BOUNDARY = 3, 
    PRECONDITIONER_FAILURE = 4 
  };
  
  struct Params {
    value_type abs_tol = value_type(1e-4);      
    value_type rel_tol = value_type(1e-2);       
    size_type max_iter = 20;                     
    bool use_preconditioner = false;              
  };
  
  struct Result {
    value_type step_norm = value_type(0);         
    value_type pred_reduction = value_type(0);    
    TerminationStatus status = TerminationStatus::CONVERGED; 
    size_type iter = 0;                           
  };

private:
  const Problem& problem_;
  clone_return_t<vector_type> r_;     // residual
  clone_return_t<vector_type> v_;     // preconditioned residual
  clone_return_t<vector_type> p_;     // search direction
  clone_return_t<vector_type> Hp_;    // H*p
  clone_return_t<vector_type> s_tmp_; // temporary step
  clone_return_t<vector_type> grad_;  // gradient storage
  
public:
  /**
   * @brief Constructor that stores reference to problem and allocates working vectors
   * @param problem Problem satisfying unconstrained_problem_c
   */
  TruncatedCG( const Problem& problem ) 
    : problem_(problem),
      r_(clone(problem.x0)),
      v_(clone(problem.x0)),
      p_(clone(problem.x0)), 
      Hp_(clone(problem.x0)),
      s_tmp_(clone(problem.x0)),
      grad_(clone(problem.x0)) {
    static_assert(objective_hess_vec_c<objective_type,vector_type>,
		  "Finite difference hessian-vector product not implemented yet");    
  }
  
  /**
   * @brief Solve trust region subproblem
   * @param s Output: computed step (input: initial guess, typically zero)
   * @param x Current iterate
   * @param delta Trust region radius
   * @param params Algorithm parameters
   * @param precond Preconditioner operator (expected to approximate H^{-1})
   * @return Result information including termination status
   */
  Result solve( vector_type& s, 
		const vector_type& x, 
		value_type delta, 
                const Params& params = {}) {
    
    const auto& obj = problem_.objective;
    auto& r = deref_if_needed(r_);
    auto& v = deref_if_needed(v_);
    auto& p = deref_if_needed(p_);
    auto& Hp = deref_if_needed(Hp_);
    auto& s_tmp = deref_if_needed(s_tmp_);
    auto& g = deref_if_needed(grad_);
    
    constexpr value_type zero(0), one(1), half(0.5);
    
    Result result;
    
    // Compute gradient at current point
    obj.gradient(g, x);
    
    // Initialize step to zero
    scale_in_place(s, zero); 
    
    value_type snorm2 = zero;
    
    // Initialize residual r = g (since s = 0 initially)
    r = g;
    value_type gnorm = rvf::sqrt(inner_product(r, r));
    value_type normg = gnorm;
    
    // Compute tolerance
    const value_type gtol = std::min(params.abs_tol, params.rel_tol * gnorm);
    
    // Apply preconditioner: v = M^{-1} * r
    if constexpr( objective_preconditioner_c<objective_type,vector_type> ) {
      if (params.use_preconditioner) {
        obj.precond(v, r, x);
      } else {
        v = r;
      }
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
        sigma = (-sMp + rvf::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
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
      value_type s1norm2 = l2norm(s_tmp);
      if (s1norm2 >= delta * delta) {
        // Step would exceed trust region
        sMp = inner_product(s, p);
        pnorm2 = inner_product(p, p);
        sigma = (-sMp + rvf::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
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
      normg = l2norm(r);
      if (normg < gtol) {
        result.step_norm = snorm2;
        result.status = TerminationStatus::CONVERGED;
        result.iter = iter + 1;
        return result;
      }
      
      if constexpr( objective_preconditioner_c<objective_type,vector_type> ) {
        if (params.use_preconditioner) {
          obj.precond(v, r, x);
        } else {
          v = r;
        }
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
    result.step_norm = rvf::sqrt(snorm2);
    result.status = TerminationStatus::MAX_ITER;
    
    return result;
  }
};

} // namespace rvf

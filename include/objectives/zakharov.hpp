/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "real_vector.hpp"
#include "algorithms/sherman_morrison.hpp"
#include <cmath>
#include <algorithm>

namespace rvf {

template<real_vector_c Vec>
using vector_value_t = inner_product_return_t<Vec>;

template<real_vector_c Vec>
using vector_size_t = dimension_return_t<Vec>;

/**
 * @brief Zakharov test function
 * 
 * Objective function:
 * f(x) = x^T x + (1/4)(k^T x)^2 + (1/16)(k^T x)^4
 * 
 * Where k = (1, 2, 3, ..., n)
 * 
 * Gradient:
 * g = ∇f(x) = 2x + (1/4)[2(k^T x) + (k^T x)^3] k
 * 
 * Hessian:
 * H = ∇²f(x) = 2I + (1/4)[2 + 3(k^T x)²] k k^T
 * 
 * The Hessian has the form H = 2I + σ k k^T, making it suitable
 * for the Sherman-Morrison formula for computing H^{-1} v.
 * 
 * Inverse Hessian action:
 * H^{-1} v = (1/2)v - (k^T v) k / (16/(2 + 3(k^T x)²) + 2(k^T k))
 */
template<real_vector_c Vec>
class zakharov_objective {
private:
  Vec k_; // Vector containing (1, 2, 3, ..., n)
  vector_value_t<Vec> k_dot_k_; // Precomputed k^T k

public:
  /**
   * @brief Constructor with provided k vector
   * @param k Index vector (1, 2, 3, ..., n)
   */
  explicit zakharov_objective(const Vec& k) 
    : k_(k), k_dot_k_(inner_product(k, k)) {}

  /**
   * @brief Compute objective function value
   * @param x Input vector
   * @return Function value f(x)
   */
  vector_value_t<Vec> value(const Vec& x) const {
    using value_type = vector_value_t<Vec>;
    
    value_type x_dot_x = inner_product(x, x);
    value_type k_dot_x = inner_product(k_, x);
    
    value_type k_dot_x_2 = k_dot_x * k_dot_x;
    value_type k_dot_x_4 = k_dot_x_2 * k_dot_x_2;
    
    return x_dot_x + value_type(0.25) * k_dot_x_2 + value_type(0.0625) * k_dot_x_4;
  }
  
  /**
   * @brief Compute gradient  
   * @param grad Output gradient vector
   * @param x Input vector
   */
  void gradient(Vec& grad, const Vec& x) const {
    using value_type = vector_value_t<Vec>;
    
    value_type k_dot_x = inner_product(k_, x);
    value_type k_dot_x_3 = k_dot_x * k_dot_x * k_dot_x;
    value_type coeff = value_type(0.25) * (value_type(2) * k_dot_x + k_dot_x_3);
    
    // grad = 2x + coeff * k
    grad = x;
    scale_in_place(grad, value_type(2));
    axpy_in_place(grad, coeff, k_);
  }
  
  /**
   * @brief Compute Hessian-vector product
   * @param hv Output: H*v
   * @param v Input vector  
   * @param x Current point
   */
  void hessVec(Vec& hv, const Vec& v, const Vec& x) const {
    using value_type = vector_value_t<Vec>;
    
    value_type k_dot_x = inner_product(k_, x);
    value_type k_dot_v = inner_product(k_, v);
    value_type coeff = value_type(0.25) * (value_type(2) + value_type(3) * k_dot_x * k_dot_x) * k_dot_v;
    
    // hv = 2v + coeff * k
    hv = v;
    scale_in_place(hv, value_type(2));
    axpy_in_place(hv, coeff, k_);
  }
  
  /**
   * @brief Compute inverse Hessian-vector product using Sherman-Morrison formula
   * @param x Current point
   * @param v Input vector
   * @param inv_hv Output: H^{-1}*v
   */
  void inv_hess_vec(const Vec& x, const Vec& v, Vec& inv_hv) const {
    using value_type = vector_value_t<Vec>;
    
    value_type k_dot_x = inner_product(k_, x);
    value_type k_dot_v = inner_product(k_, v);
    
    // For H = 2I + σ k k^T where σ = (1/4)[2 + 3(k^T x)²]
    // Using Sherman-Morrison: (2I + σ k k^T)^{-1} = (1/2)I - (σ/2) k k^T / (1 + (σ/2) k^T k)
    // Simplifying: H^{-1} v = (1/2)v - (k^T v) k / (4/σ + 2 k^T k)
    
    value_type sigma = value_type(0.25) * (value_type(2) + value_type(3) * k_dot_x * k_dot_x);
    value_type denominator = value_type(4) / sigma + value_type(2) * k_dot_k_;
    value_type coeff = -k_dot_v / denominator;
    
    // inv_hv = (1/2) * v + coeff * k
    inv_hv = v;
    scale_in_place(inv_hv, value_type(0.5));
    axpy_in_place(inv_hv, coeff, k_);
  }
  
  /**
   * @brief Alternative inverse Hessian using general Sherman-Morrison
   * @param x Current point
   * @param v Input vector
   * @param inv_hv Output: H^{-1}*v
   */
  void inv_hess_vec_sherman_morrison(const Vec& x, const Vec& v, Vec& inv_hv) const {
    using value_type = vector_value_t<Vec>;
    
    value_type k_dot_x = inner_product(k_, x);
    value_type sigma = value_type(0.25) * (value_type(2) + value_type(3) * k_dot_x * k_dot_x);
    
    // H = 2I + σ k k^T
    // We need to solve (2I + σ k k^T) inv_hv = v
    // This is in the form (A + u v^T) x = b where:
    // A = 2I, u = √σ k, v = √σ k, b = v
    
    // Create scaled k vectors for Sherman-Morrison
    auto u_cl = clone(k_); auto& u = deref_if_needed(u_cl);
    auto v_sm_cl = clone(k_); auto& v_sm = deref_if_needed(v_sm_cl);
    
    value_type sqrt_sigma = std::sqrt(sigma);
    scale_in_place(u, sqrt_sigma);
    scale_in_place(v_sm, sqrt_sigma);
    
    // Define A^{-1} operation: (2I)^{-1} w = (1/2) w
    auto A_inv = [](Vec& y, const Vec& w) {
      y = w;
      scale_in_place(y, vector_value_t<Vec>(0.5));
    };
    
    sherman_morrison_general(A_inv, u, v_sm, v, inv_hv);
  }
  
  /**
   * @brief Get the k vector (for testing purposes)
   */
  const Vec& get_k() const { return k_; }
  
  /**
   * @brief Get precomputed k^T k value
   */
  vector_value_t<Vec> get_k_dot_k() const { return k_dot_k_; }

};

/**
 * @brief Factory function to create Zakharov objective with provided k vector
 */
template<real_vector_c Vec>
zakharov_objective<Vec> make_zakharov_objective(const Vec& k) {
  return zakharov_objective<Vec>(k);
}

/**
 * @brief Factory function to create Zakharov objective with automatic k vector
 * @param x_template Template vector to determine dimension and structure
 */
template<real_vector_c Vec>
zakharov_objective<Vec> make_zakharov_objective_auto_k(const Vec& x_template) {
  auto k_clone = clone(x_template); 
  auto& k = deref_if_needed(k_clone);
  
  auto dim = dimension(x_template);
  
  // Create index vector k = (1, 2, 3, ..., n)
  // This requires specialization for each Vec type
  create_index_vector_impl(k, dim);
  
  return zakharov_objective<Vec>(k);
}

/**
 * @brief Create index vector k = (1, 2, 3, ..., n)
 * This function must be specialized for each Vec type
 */
template<real_vector_c Vec>
void create_index_vector_impl(Vec& k, vector_size_t<Vec> n) {
  // This is a placeholder that must be specialized for each Vec type
  static_assert(sizeof(Vec) == 0, 
    "create_index_vector_impl must be specialized for your Vec type");
}

} // namespace rvf

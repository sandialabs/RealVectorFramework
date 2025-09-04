/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include <cmath>
#include <limits>

namespace rvf {


// Container of vectors concept for multiple RHS; element type must be Vec
template<typename Container, typename Vec>
concept vector_container_of_c = requires(Container& container) {
  typename Container::value_type;
  requires std::same_as<typename Container::value_type, Vec>;
  { container.size() } -> std::convertible_to<std::size_t>;
  { container[0] } -> std::convertible_to<Vec&>;
};

/**
 * @brief Solves the linear system (A + u*v^T)x = b using the Sherman-Morrison formula.
 * 
 * This algorithm is efficient when A is easy to invert (like identity matrix)
 * and we need to solve a system with a rank-1 update A + u*v^T.
 * 
 * The Sherman-Morrison formula: (A + uv^T)^(-1) = A^(-1) - (A^(-1)uv^TA^(-1))/(1 + v^TA^(-1)u)
 * 
 * For the case where A = I (identity matrix), this simplifies to:
 * x = b - u*(v^T*b)/(1 + v^T*u)
 */
template<real_vector_c Vec>
void sherman_morrison_identity_plus_rank1(
    const Vec& u,    // rank-1 update vector u
    const Vec& v,    // rank-1 update vector v  
    const Vec& b,    // right-hand side
    Vec& x           // solution vector (output)
) {
    using value_t = inner_product_return_t<Vec>;
    
    // For A = I, the solution is: x = b - u*(v^T*b)/(1 + v^T*u)
    
    // First, copy b to x (keep owner and reference adjacent)
    auto b_clone = clone(b); x = deref_if_needed(b_clone);
    
    // Compute v^T * b
    value_t vtb = inner_product(v, b);
    
    // Compute v^T * u  
    value_t vtu = inner_product(v, u);
    
    // Check for singularity: if (1 + v^T*u) ≈ 0, the system is singular
    value_t denominator = static_cast<value_t>(1) + vtu;
    if (std::abs(denominator) < std::numeric_limits<value_t>::epsilon()) {
        // System is singular or nearly singular
        return;
    }
    
    // Compute the scalar coefficient: -(v^T*b)/(1 + v^T*u)
    value_t coeff = -vtb / denominator;
    
    // Create a scaled copy of u
    auto u_scaled = clone(u); auto& u_scaled_ref = deref_if_needed(u_scaled);
    scale_in_place(u_scaled_ref, coeff);
    
    // x = b + coeff * u (where coeff is negative, so this subtracts)
    add_in_place(x, u_scaled_ref);
}

/**
 * @brief General Sherman-Morrison solver for (A + u*v^T)x = b when A^(-1) is known
 * 
 * @param A_inv A function/functor that computes A^(-1) * y for any vector y (expected to be linear)
 * @param u rank-1 update vector u
 * @param v rank-1 update vector v
 * @param b right-hand side vector
 * @param x solution vector (output)
 */
template<typename InverseOperator, real_vector_c Vec>
requires self_map_c<InverseOperator, Vec>
void sherman_morrison_general(
    const InverseOperator& A_inv,
    const Vec& u,
    const Vec& v, 
    const Vec& b,
    Vec& x
) {
    using value_t = inner_product_return_t<Vec>;
    
    // Step 1: Compute A^(-1) * b
    auto b_cl = clone(b); auto& Ainv_b = deref_if_needed(b_cl);
    A_inv(deref_if_needed(Ainv_b), b);
    
    // Step 2: Compute A^(-1) * u
    auto u_cl = clone(u); auto& Ainv_u = deref_if_needed(u_cl);
    A_inv(deref_if_needed(Ainv_u), u);
    
    // Step 3: Compute v^T * A^(-1) * u
    value_t vT_Ainv_u = inner_product(v, deref_if_needed(Ainv_u));
    
    // Step 4: Check for singularity
    value_t denominator = static_cast<value_t>(1) + vT_Ainv_u;
    if (std::abs(denominator) < std::numeric_limits<value_t>::epsilon()) {
        // System is singular, return A^(-1) * b as best approximation
        x = deref_if_needed(Ainv_b);
        return;
    }
    
    // Step 5: Compute v^T * A^(-1) * b
    value_t vT_Ainv_b = inner_product(v, deref_if_needed(Ainv_b));
    
    // Step 6: Compute the correction coefficient
    value_t coeff = -vT_Ainv_b / denominator;
    
    // Step 7: Apply Sherman-Morrison formula
    // x = A^(-1)*b - (A^(-1)*u * v^T * A^(-1)*b) / (1 + v^T * A^(-1)*u)
    x = deref_if_needed(Ainv_b);  // Start with A^(-1) * b
    
    // Create scaled A^(-1)*u
    auto Ainv_u_cl = clone(Ainv_u); auto& correction = deref_if_needed(Ainv_u_cl);
    auto& correction_ref = deref_if_needed(correction);
    scale_in_place(correction_ref, coeff);
    
    // Add the correction term
    add_in_place(x, correction_ref);
}

/**
 * @brief Solves multiple right-hand sides simultaneously using Sherman-Morrison
 * 
 * This is efficient when you need to solve (I + u*v^T)X = B where B has multiple columns.
 */
template<real_vector_c Vec, typename Container>
requires vector_container_of_c<Container, Vec>
void sherman_morrison_multiple_rhs(
    const Vec& u,
    const Vec& v,
    const Container& B,  // Multiple right-hand sides
    Container& X         // Multiple solutions (output)
) {
    using value_t = inner_product_return_t<Vec>;
    
    // Precompute v^T * u once (same for all right-hand sides)
    value_t vtu = inner_product(v, u);
    value_t denominator = static_cast<value_t>(1) + vtu;
    
    if (std::abs(denominator) < std::numeric_limits<value_t>::epsilon()) {
        // Singular case - just copy B to X
        for (std::size_t i = 0; i < B.size(); ++i) { X[i] = B[i]; }
        return;
    }
    
    // Solve each system
    for (std::size_t i = 0; i < B.size(); ++i) {
        // x_i = b_i - u * (v^T * b_i) / (1 + v^T * u)
        auto b_clone = clone(B[i]); X[i] = deref_if_needed(b_clone);
        
        value_t vtb = inner_product(v, B[i]);
        value_t coeff = -vtb / denominator;
        
        auto u_scaled = clone(u); auto& u_scaled_ref = deref_if_needed(u_scaled);
        scale_in_place(u_scaled_ref, coeff);
        
        add_in_place(X[i], u_scaled_ref);
    }
}

} // namespace rvf

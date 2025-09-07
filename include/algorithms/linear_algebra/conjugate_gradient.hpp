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
#include <cmath>
#include <algorithm> // For std::max

namespace rvf {



template<real_vector_c Vec>
using vector_value_t = inner_product_return_t<Vec>;

template<real_vector_c Vec>
using vector_size_t = dimension_return_t<Vec>;

/**
 * @brief Solves the linear system Ax = b using the Conjugate Gradient method.
 * 
 * @param A Linear operator (expected to be symmetric positive definite)
 */
template <typename Matrix, real_vector_c Vec>
requires self_map_c<Matrix, Vec>
void conjugate_gradient( const Matrix& A,
                         const Vec& b,
                         Vec& x,
                         vector_value_t<Vec> relTol = 1e-5,
                         vector_value_t<Vec> absTol = 0,
                         vector_size_t<Vec> maxIter = 100 ) {

  auto tol = std::max(relTol * rvf::sqrt(inner_product(b, b)), absTol);
  auto b_cl = clone(b); auto& r = deref_if_needed(b_cl);

  A(r, x);
  scale_in_place(r, -1.0);
  add_in_place(r, b);

  auto rho0 = inner_product(r, r);
  if(rvf::sqrt(rho0) < tol) return;

  auto r_cl = clone(r); auto& p  = deref_if_needed(r_cl); 
  auto x_cl = clone(x); auto& Ap = deref_if_needed(x_cl); 

  for(vector_size_t<Vec> iter = 0; iter < maxIter; ++iter) {
    A(Ap, p);
    auto pAp = inner_product(Ap, p);
    auto alpha = rho0 / pAp;
    axpy_in_place(x,  alpha,  p);
    axpy_in_place(r, -alpha, Ap);
    auto rho = inner_product(r, r);
    if(rvf::sqrt(rho) < tol) break;
    auto beta = rho / rho0;
    scale_in_place(p, beta);
    add_in_place(p, r);
    rho0 = rho;
  }
}

} // namespace rvf

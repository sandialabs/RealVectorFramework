/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "../real_vector.hpp"
#include "../operations/axpy_in_place.hpp"
#include <cmath>
#include <algorithm> // For std::max

namespace rvf {

// Linear operator concept: A(y, x) performs y = A*x and returns void
template<typename M, typename V>
concept linear_operator_c = requires(const M& A, V& y, const V& x) {
  { A(y, x) } -> std::same_as<void>;
};


template<real_vector_c Vec>
using vector_value_t = inner_product_return_t<Vec>;

template<real_vector_c Vec>
using vector_size_t = dimension_return_t<Vec>;

/**
 * @brief Solves the linear system Ax = b using the Conjugate Gradient method.
 */
template <typename Matrix, real_vector_c Vec>
requires linear_operator_c<Matrix, Vec>
void conjugate_gradient(
    const Matrix& A,
    const Vec& b,
    Vec& x,
    vector_value_t<Vec> relTol = 1e-5,
    vector_value_t<Vec> absTol = 0,
    vector_size_t<Vec> maxIter = 100)
{
    auto tol = std::max(relTol * std::sqrt(inner_product(b, b)), absTol);

    auto r = clone(b);
    A(r, x);
    scale_in_place(r, -1.0);
    add_in_place(r, b);

    auto rho0 = inner_product(r, r);
    if (std::sqrt(rho0) < tol) {
        return;
    }

    auto p = clone(r);
    auto Ap = clone(x);

    for (vector_size_t<Vec> iter = 0; iter < maxIter; ++iter) {
        A(Ap, p);
        auto pAp = inner_product(Ap, p);
        auto alpha = rho0 / pAp;
        axpy_in_place(x, alpha, p);
        axpy_in_place(r, -alpha, Ap);
        auto rho = inner_product(r, r);
        if (std::sqrt(rho) < tol) {
            break;
        }
        auto beta = rho / rho0;
        scale_in_place(p, beta);
        add_in_place(p, r);
        rho0 = rho;
    }
}

} // namespace rvf

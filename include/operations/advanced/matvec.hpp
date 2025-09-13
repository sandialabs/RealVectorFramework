/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once
#include <tincup/tincup.hpp>

namespace rvf {

/**
 * @brief Matrix-vector multiplication: y = A * x
 * @cpo
 * Computes the matrix-vector product y = A * x, where A is represented
 * as a container of vectors (rows), x is an input vector, and y is the
 * output vector that receives the result.
 * 
 * This operation is fundamental in linear algebra and neural network
 * computations, particularly for linear transformations and dense layers.
 * 
 * The matrix A is expected to be iterable, where each element represents
 * a row vector. The computation performs: y[i] = A[i] · x for each row i.
 *
 * @param y Output vector (modified in-place)
 * @param A Matrix (container of row vectors)
 * @param x Input vector
 *
 * @cpo_example
 * @code
 * std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};  // 2x2 matrix
 * std::vector<double> x = {1, 1};  // Input vector
 * std::vector<double> y(2);        // Output vector
 * matvec(y, A, x);                 // y becomes {3, 7}
 * @endcode
 */
inline constexpr struct matvec_ftor final : tincup::cpo_base<matvec_ftor> {
  TINCUP_CPO_TAG("matvec")
  inline static constexpr bool is_variadic = false;
  
  template<typename V, typename M, typename X>
    requires tincup::invocable_c<matvec_ftor, V&, const M&, const X&>
  constexpr auto operator()(V& y, const M& A, const X& x) const
    noexcept(tincup::nothrow_invocable_c<matvec_ftor, V&, const M&, const X&>) 
    -> tincup::invocable_t<matvec_ftor, V&, const M&, const X&> {
    return tincup::tag_invoke_cpo(*this, y, A, x);
  }
} matvec;

// CPO-specific concepts and type aliases for convenient usage
template<typename V, typename M, typename X>
concept matvec_invocable_c = tincup::invocable_c<matvec_ftor, V&, const M&, const X&>;

template<typename V, typename M, typename X>
concept matvec_nothrow_invocable_c = tincup::nothrow_invocable_c<matvec_ftor, V&, const M&, const X&>;

template<typename V, typename M, typename X>
using matvec_return_t = tincup::invocable_t<matvec_ftor, V&, const M&, const X&>;

template<typename V, typename M, typename X>
using matvec_traits = tincup::cpo_traits<matvec_ftor, V&, const M&, const X&>;

} // namespace rvf
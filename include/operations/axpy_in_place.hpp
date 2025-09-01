/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "../real_vector.hpp"

namespace rvf {

// NOTE: To make @cpo, @cpo_example, and @tag_invoke_impl recognized by Doxygen,
// add these aliases to your Doxyfile:
//   ALIASES += cpo="Customization Point Object"
//   ALIASES += cpo_example="Example usage of CPO"
//   ALIASES += tag_invoke_impl="Default tag_invoke overload implementation"
/**
 * @brief Performs the AXPY operation: y = alpha * x + y (in-place)
 * @cpo
 * Computes the Alpha X Plus Y operation in-place on the target vector.
 * This is a fundamental BLAS Level 1 operation that scales vector x by 
 * scalar alpha and adds the result to vector y, storing the result in y.
 *
 * @cpo_example
 * @code
 * axpy_in_place(y, 2.5, x);  // y = 2.5*x + y
 * @endcode
 */

inline constexpr struct axpy_in_place_ftor final : tincup::cpo_base<axpy_in_place_ftor> {
  TINCUP_CPO_TAG("axpy_in_place")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename S, typename V>
    requires tincup::invocable_c<axpy_in_place_ftor, V&, S, const V&>
  constexpr auto operator()(V& y, S alpha, const V& x) const
    noexcept(tincup::nothrow_invocable_c<axpy_in_place_ftor, V&, S, const V&>) 
    -> tincup::invocable_t<axpy_in_place_ftor, V&, S, const V&> {
    return tincup::tag_invoke_cpo(*this, y, alpha, x);
  }
} axpy_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename S, typename V>
concept axpy_in_place_invocable_c = tincup::invocable_c<axpy_in_place_ftor, V&, S, const V&>;

template<typename S, typename V>
concept axpy_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<axpy_in_place_ftor, V&, S, const V&>;


template<typename S, typename V>
using axpy_in_place_return_t = tincup::invocable_t<axpy_in_place_ftor, V&, S, const V&>;

template<typename S, typename V>
using axpy_in_place_traits = tincup::cpo_traits<axpy_in_place_ftor, V&, S, const V&>;

// Usage: tincup::is_invocable_v<axpy_in_place_ftor, V&, S, const V&>


/**
 * @brief Default tag_invoke overload for axpy_in_place
 * @tag_invoke_impl
 * Provides a default implementation that performs y = alpha * x + y
 * by cloning x, scaling the clone by alpha, then adding to y.
 *
 * @param y Target vector (modified in-place)
 * @param alpha Scalar multiplier for vector x
 * @param x Source vector (not modified)
 * @return void
 */
template<typename S, typename V>
constexpr auto tag_invoke(axpy_in_place_ftor, V& y, S alpha, const V& x) -> void {
    // Standard AXPY implementation using existing CPOs
    auto x_clone = clone(x);
    auto& alphax = deref_if_needed(x_clone);
    scale_in_place(alphax, alpha);
    add_in_place(y, alphax);
}

} // namespace rvf

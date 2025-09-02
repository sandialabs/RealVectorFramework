/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once
#include <tincup/tincup.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "deref_if_needed.hpp"

namespace rvf {

// NOTE: To make @cpo, @cpo_example, and @tag_invoke_impl recognized by Doxygen,
// add these aliases to your Doxyfile:
//   ALIASES += cpo="Customization Point Object"
//   ALIASES += cpo_example="Example usage of CPO"
//   ALIASES += tag_invoke_impl="Default tag_invoke overload implementation"
/**
 * @brief Makes a by-value copy of a vector
 * @cpo
 * @ingroup tincup_cpos
 * Returns a copy of the input vector-like object. Implementations may return
 * a wrapper that dereferences to a vector; use `rvf::deref_if_needed` to
 * obtain a reference when composing operations.
 *
 * Recommended idiom: keep the owner and reference adjacent to avoid
 * dangling references and to satisfy concept-constrained APIs that
 * check types before conversions.
 *
 * @code
 * auto cl = rvf::clone(x); auto& xr = rvf::deref_if_needed(cl);
 * // use xr as a regular vector, cl keeps storage alive
 * @endcode
 *
 * Note: template argument deduction does not consider user-defined
 * conversions, so passing temporary wrappers to functions constrained
 * on `real_vector_c` will fail. Always bind a reference via the idiom above.
 *
 * @cpo_example
 * @code
 * std::vector<double> x{1,2,3};
 * auto xc = rvf::clone(x); // copy of x
 * @endcode
 */

inline constexpr struct clone_ftor final : tincup::cpo_base<clone_ftor> {
  TINCUP_CPO_TAG("clone")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename V>
    requires tincup::invocable_c<clone_ftor, const V&>
  constexpr auto operator()(const V& x) const
    noexcept(tincup::nothrow_invocable_c<clone_ftor, const V&>) 
    -> tincup::invocable_t<clone_ftor, const V&> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} clone;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept clone_invocable_c = tincup::invocable_c<clone_ftor, const V&>;

template<typename V>
concept clone_nothrow_invocable_c = tincup::nothrow_invocable_c<clone_ftor, const V&>;


template<typename V>
using clone_return_t = tincup::invocable_t<clone_ftor, const V&>;

template<typename V>
using clone_traits = tincup::cpo_traits<clone_ftor, const V&>;

// Usage: tincup::is_invocable_v<clone_ftor, const V&>


// No unconstrained default tag_invoke. See std_ranges_support.hpp or provide
// a user-defined overload via ADL for your types.

} // namespace rvf

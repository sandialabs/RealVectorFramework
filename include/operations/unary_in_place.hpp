/**
TInCuP - A library for generating and validating C++ customization point objects that use `tag_invoke`

Copyright (c) National Technology & Engineering Solutions of Sandia, 
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once
#include <tincup/tincup.hpp>

namespace rvf {

// NOTE: To make @cpo, @cpo_example, and @tag_invoke_impl recognized by Doxygen,
// add these aliases to your Doxyfile:
//   ALIASES += cpo="Customization Point Object"
//   ALIASES += cpo_example="Example usage of CPO"
//   ALIASES += tag_invoke_impl="Default tag_invoke overload implementation"
/**
 * @brief Applies a unary function to each element of a vector in-place
 * @cpo
 * Applies the given callable object to each element of the target vector,
 * modifying the vector in place. The callable must be able to take an
 * argument of inner_product_return_t<V> and return a value of the same type.
 *
 * @cpo_example
 * @code
 * unary_in_place(vec, [](auto x) { return x * x; });  // Square each element
 * @endcode
 */

inline constexpr struct unary_in_place_ftor final : tincup::cpo_base<unary_in_place_ftor> {
  TINCUP_CPO_TAG("unary_in_place")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename F, typename V>
    requires tincup::invocable_c<unary_in_place_ftor, V&, F>
  constexpr auto operator()(V& target, F func) const
    noexcept(tincup::nothrow_invocable_c<unary_in_place_ftor, V&, F>) 
    -> tincup::invocable_t<unary_in_place_ftor, V&, F> {
    return tincup::tag_invoke_cpo(*this, target, func);
  }
} unary_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename F, typename V>
concept unary_in_place_invocable_c = tincup::invocable_c<unary_in_place_ftor, V&, F>;

template<typename F, typename V>
concept unary_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<unary_in_place_ftor, V&, F>;


template<typename F, typename V>
using unary_in_place_return_t = tincup::invocable_t<unary_in_place_ftor, V&, F>;

template<typename F, typename V>
using unary_in_place_traits = tincup::cpo_traits<unary_in_place_ftor, V&, F>;

// Usage: tincup::is_invocable_v<unary_in_place_ftor, V&, F>


// No unconstrained default tag_invoke. See std_ranges_support.hpp or provide
// a user-defined overload via ADL for your types.

} // namespace rvf

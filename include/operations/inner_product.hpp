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
 * @brief [TODO: Brief description for inner_product]
 * @cpo
 * @ingroup tincup_cpos
 * [TODO: Detailed description of the CPO.]
 *
 * @cpo_example
 * @code
 * auto result = inner_product( args... );
 * @endcode
 */

inline constexpr struct inner_product_ftor final : tincup::cpo_base<inner_product_ftor> {
  TINCUP_CPO_TAG("inner_product")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename V>
    requires tincup::invocable_c<inner_product_ftor, const V&, const V&>
  constexpr auto operator()(const V& x, const V& y) const
    noexcept(tincup::nothrow_invocable_c<inner_product_ftor, const V&, const V&>) 
    -> tincup::invocable_t<inner_product_ftor, const V&, const V&> {
    return tincup::tag_invoke_cpo(*this, x, y);
  }
} inner_product;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept inner_product_invocable_c = tincup::invocable_c<inner_product_ftor, const V&, const V&>;

template<typename V>
concept inner_product_nothrow_invocable_c = tincup::nothrow_invocable_c<inner_product_ftor, const V&, const V&>;


template<typename V>
using inner_product_return_t = tincup::invocable_t<inner_product_ftor, const V&, const V&>;

template<typename V>
using inner_product_traits = tincup::cpo_traits<inner_product_ftor, const V&, const V&>;

// Usage: tincup::is_invocable_v<inner_product_ftor, const V&, const V&>


// No unconstrained default tag_invoke. See std_ranges_support.hpp or provide
// a user-defined overload via ADL for your types.

} // namespace rvf

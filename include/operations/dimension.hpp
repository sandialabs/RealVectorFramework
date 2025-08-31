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
 * @brief [TODO: Brief description for dimension]
 * @cpo
 * @ingroup tincup_cpos
 * [TODO: Detailed description of the CPO.]
 *
 * @cpo_example
 * @code
 * auto result = dimension( args... );
 * @endcode
 */

inline constexpr struct dimension_ftor final : tincup::cpo_base<dimension_ftor> {
  TINCUP_CPO_TAG("dimension")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename V>
    requires tincup::invocable_c<dimension_ftor, const V&>
  constexpr auto operator()(const V& x) const
    noexcept(tincup::nothrow_invocable_c<dimension_ftor, const V&>) 
    -> tincup::invocable_t<dimension_ftor, const V&> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} dimension;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept dimension_invocable_c = tincup::invocable_c<dimension_ftor, const V&>;

template<typename V>
concept dimension_nothrow_invocable_c = tincup::nothrow_invocable_c<dimension_ftor, const V&>;


template<typename V>
using dimension_return_t = tincup::invocable_t<dimension_ftor, const V&>;

template<typename V>
using dimension_traits = tincup::cpo_traits<dimension_ftor, const V&>;

// Usage: tincup::is_invocable_v<dimension_ftor, const V&>


// No unconstrained default tag_invoke. See std_ranges_support.hpp or provide
// a user-defined overload via ADL for your types.

} // namespace rvf

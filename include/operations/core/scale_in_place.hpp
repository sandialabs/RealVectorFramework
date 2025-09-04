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

// NOTE: To make @cpo, @cpo_example, and @tag_invoke_impl recognized by Doxygen,
// add these aliases to your Doxyfile:
//   ALIASES += cpo="Customization Point Object"
//   ALIASES += cpo_example="Example usage of CPO"
//   ALIASES += tag_invoke_impl="Default tag_invoke overload implementation"
/**
 * @brief Elementwise x *= alpha (in-place)
 * @cpo
 * @ingroup tincup_cpos
 * Multiplies each element of the target vector by the scalar alpha, modifying
 * the vector in place. The scalar type should match the vector's value type.
 *
 * @cpo_example
 * @code
 * std::vector<double> v{1,2,3};
 * rvf::scale_in_place(v, 2.0); // v == {2,4,6}
 * @endcode
 */

inline constexpr struct scale_in_place_ftor final : tincup::cpo_base<scale_in_place_ftor> {
  TINCUP_CPO_TAG("scale_in_place")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename S, typename V>
    requires tincup::invocable_c<scale_in_place_ftor, V&, S>
  constexpr auto operator()(V& x, S alpha) const
    noexcept(tincup::nothrow_invocable_c<scale_in_place_ftor, V&, S>) 
    -> tincup::invocable_t<scale_in_place_ftor, V&, S> {
    return tincup::tag_invoke_cpo(*this, x, alpha);
  }
} scale_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename S, typename V>
concept scale_in_place_invocable_c = tincup::invocable_c<scale_in_place_ftor, V&, S>;

template<typename S, typename V>
concept scale_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<scale_in_place_ftor, V&, S>;


template<typename S, typename V>
using scale_in_place_return_t = tincup::invocable_t<scale_in_place_ftor, V&, S>;

template<typename S, typename V>
using scale_in_place_traits = tincup::cpo_traits<scale_in_place_ftor, V&, S>;

// Usage: tincup::is_invocable_v<scale_in_place_ftor, V&, S>


// No unconstrained default tag_invoke. See std_ranges_support.hpp or provide
// a user-defined overload via ADL for your types.

} // namespace rvf

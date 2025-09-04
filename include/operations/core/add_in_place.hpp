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
 * @brief Elementwise y += x (in-place)
 * @cpo
 * @ingroup tincup_cpos
 * Adds each element of the source vector to the corresponding element of the
 * target vector, writing the result into the target. Implementations may be
 * provided via ADL `tag_invoke` or `tincup::cpo_impl` specializations.
 *
 * @cpo_example
 * @code
 * std::vector<double> x{1,2,3}, y{4,5,6};
 * rvf::add_in_place(y, x); // y == {5,7,9}
 * @endcode
 */

inline constexpr struct add_in_place_ftor final : tincup::cpo_base<add_in_place_ftor> {
  TINCUP_CPO_TAG("add_in_place")
  inline static constexpr bool is_variadic = false;
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename T, typename U>
    requires tincup::invocable_c<add_in_place_ftor, T&, const U&>
  constexpr auto operator()(T& target, const U& source) const
    noexcept(tincup::nothrow_invocable_c<add_in_place_ftor, T&, const U&>) 
    -> tincup::invocable_t<add_in_place_ftor, T&, const U&> {
    return tincup::tag_invoke_cpo(*this, target, source);
  }
} add_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename T, typename U>
concept add_in_place_invocable_c = tincup::invocable_c<add_in_place_ftor, T&, const U&>;

template<typename T, typename U>
concept add_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<add_in_place_ftor, T&, const U&>;

// Enhanced semantic concept with meaningful requirements
template<typename T, typename U>
concept add_in_place_c = add_in_place_invocable_c<T&, const U&> && requires {
    requires std::is_move_constructible_v<T>;
    requires std::is_void_v<tincup::invocable_t<add_in_place_ftor, T&, const U&>>;
};

template<typename T, typename U>
using add_in_place_return_t = tincup::invocable_t<add_in_place_ftor, T&, const U&>;

template<typename T, typename U>
using add_in_place_traits = tincup::cpo_traits<add_in_place_ftor, T&, const U&>;

// Usage: tincup::is_invocable_v<add_in_place_ftor, T&, const U&>
// Usage: add_in_place_c<T&, const U&> (semantic requirements included)


// Intentionally no unconstrained default tag_invoke here.
// Implementations are provided for supported types (e.g., std::ranges) in
// operations/std_ranges_support.hpp or by user-defined overloads via ADL.

} // namespace rvf

/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <tincup/tincup.hpp>
#include "core/real_scalar.hpp"

namespace rvf {

inline constexpr struct sqrt_ftor final : tincup::cpo_base<sqrt_ftor> {
  TINCUP_CPO_TAG("sqrt")
  inline static constexpr bool is_variadic = false;
  template<typename T>
  requires tincup::invocable_c<sqrt_ftor, T>
  constexpr auto operator()(T x) const
  noexcept(tincup::nothrow_invocable_c<sqrt_ftor, T>) 
  -> tincup::invocable_t<sqrt_ftor, T> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} sqrt;

template<typename T>
concept sqrt_invocable_c = tincup::invocable_c<sqrt_ftor, T>;

template<typename T>
concept sqrt_nothrow_invocable_c = tincup::nothrow_invocable_c<sqrt_ftor, T>;


template<typename T>
using sqrt_return_t = tincup::invocable_t<sqrt_ftor, T>;

template<typename T>
using sqrt_traits = tincup::cpo_traits<sqrt_ftor, T>;

// Usage: tincup::is_invocable_v<sqrt_ftor, T>


} // namespace rvf

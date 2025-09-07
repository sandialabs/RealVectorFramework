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

inline constexpr struct pow_ftor final : tincup::cpo_base<pow_ftor> {
  TINCUP_CPO_TAG("pow")
  inline static constexpr bool is_variadic = false;
  template<typename T>
  requires tincup::invocable_c<pow_ftor, T, T>
  constexpr auto operator()(T x, T y) const
  noexcept(tincup::nothrow_invocable_c<pow_ftor, T, T>) 
  -> tincup::invocable_t<pow_ftor, T, T> {
    return tincup::tag_invoke_cpo(*this, x, y);
  }
} pow;

template<typename T>
concept pow_invocable_c = tincup::invocable_c<pow_ftor, T, T>;

template<typename T>
concept pow_nothrow_invocable_c = tincup::nothrow_invocable_c<pow_ftor, T, T>;


template<typename T>
using pow_return_t = tincup::invocable_t<pow_ftor, T, T>;

template<typename T>
using pow_traits = tincup::cpo_traits<pow_ftor, T, T>;

// Usage: tincup::is_invocable_v<pow_ftor, T, T>


} // namespace rvf

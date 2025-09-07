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

inline constexpr struct abs_ftor final : tincup::cpo_base<abs_ftor> {
  TINCUP_CPO_TAG("abs")
  inline static constexpr bool is_variadic = false;

  template<real_scalar_c T>
  requires tincup::invocable_c<abs_ftor, T>
  constexpr auto operator()(T x) const
  noexcept(tincup::nothrow_invocable_c<abs_ftor, T>) 
  -> tincup::invocable_t<abs_ftor, T> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} abs;

template<typename T>
concept abs_invocable_c = tincup::invocable_c<abs_ftor, T>;

template<typename T>
concept abs_nothrow_invocable_c = tincup::nothrow_invocable_c<abs_ftor, T>;

template<typename T>
using abs_return_t = tincup::invocable_t<abs_ftor, T>;

template<typename T>
using abs_traits = tincup::cpo_traits<abs_ftor, T>;

} // namespace rvf

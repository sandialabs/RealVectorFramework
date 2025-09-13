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

inline constexpr struct exp_ftor final : tincup::cpo_base<exp_ftor> {
  TINCUP_CPO_TAG("exp")
  inline static constexpr bool is_variadic = false;

  template<real_scalar_c T>
  requires tincup::invocable_c<exp_ftor, T>
  constexpr auto operator()(T x) const
  noexcept(tincup::nothrow_invocable_c<exp_ftor, T>) 
  -> tincup::invocable_t<exp_ftor, T> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} exp;

template<typename T>
concept exp_invocable_c = tincup::invocable_c<exp_ftor, T>;

template<typename T>
concept exp_nothrow_invocable_c = tincup::nothrow_invocable_c<exp_ftor, T>;

template<typename T>
using exp_return_t = tincup::invocable_t<exp_ftor, T>;

template<typename T>
using exp_traits = tincup::cpo_traits<exp_ftor, T>;

} // namespace rvf

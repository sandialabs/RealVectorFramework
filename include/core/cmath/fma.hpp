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

inline constexpr struct fma_ftor final : tincup::cpo_base<fma_ftor> {
  TINCUP_CPO_TAG("fma")
  inline static constexpr bool is_variadic = false;
  template<typename T>
  requires tincup::invocable_c<fma_ftor, T, T, T>
  constexpr auto operator()(T x, T y, T z) const
  noexcept(tincup::nothrow_invocable_c<fma_ftor, T, T, T>) 
  -> tincup::invocable_t<fma_ftor, T, T, T> {
    return tincup::tag_invoke_cpo(*this, x, y, z);
  }
} fma;

template<typename T>
concept fma_invocable_c = tincup::invocable_c<fma_ftor, T, T, T>;

template<typename T>
concept fma_nothrow_invocable_c = tincup::nothrow_invocable_c<fma_ftor, T, T, T>;

template<typename T>
using fma_return_t = tincup::invocable_t<fma_ftor, T, T, T>;

template<typename T>
using fma_traits = tincup::cpo_traits<fma_ftor, T, T, T>;

} // namespace rvf

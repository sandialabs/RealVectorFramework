#pragma once

#include <tincup/tincup.hpp>
#include "core/real_scalar.hpp"

namespace rvf {

	
inline constexpr struct fmax_ftor final : tincup::cpo_base<fmax_ftor> {
  TINCUP_CPO_TAG("fmax")
  inline static constexpr bool is_variadic = false;
  template<typename T>
  requires tincup::invocable_c<fmax_ftor, T, T>
  constexpr auto operator()(T x, T y) const
  noexcept(tincup::nothrow_invocable_c<fmax_ftor, T, T>) 
  -> tincup::invocable_t<fmax_ftor, T, T> {
    return tincup::tag_invoke_cpo(*this, x, y);
  }
} fmax;

template<typename T>
concept fmax_invocable_c = tincup::invocable_c<fmax_ftor, T, T>;

template<typename T>
concept fmax_nothrow_invocable_c = tincup::nothrow_invocable_c<fmax_ftor, T, T>;

template<typename T>
using fmax_return_t = tincup::invocable_t<fmax_ftor, T, T>;

template<typename T>
using fmax_traits = tincup::cpo_traits<fmax_ftor, T, T>;

} // namespace rvf

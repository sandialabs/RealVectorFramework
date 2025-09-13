#pragma once

#include <tincup/tincup.hpp>
#include "core/real_vector.hpp"

namespace rvf {

inline constexpr struct fill_ftor final : tincup::cpo_base<fill_ftor> {
  TINCUP_CPO_TAG("fill")
  inline static constexpr bool is_variadic = false;

  template<typename V, typename T>
    requires tincup::invocable_c<fill_ftor, V&, const T&>
  constexpr auto operator()(V& v, const T& value) const
    noexcept(tincup::nothrow_invocable_c<fill_ftor, V&, const T&>)
    -> tincup::invocable_t<fill_ftor, V&, const T&> {
    return tincup::tag_invoke_cpo(*this, v, value);
  }
} fill;

} // namespace rvf

#pragma once

#include <tincup/tincup.hpp>
#include "core/real_vector.hpp"

namespace rvf {

// Element-wise assignment for the common prefix of two vectors/ranges.
// Semantics: for i in [0, min(dim(y), dim(x))): y[i] = x[i]
// Does not resize; copies only over the overlapping region.
inline constexpr struct assign_ftor final : tincup::cpo_base<assign_ftor> {
  TINCUP_CPO_TAG("assign")
  inline static constexpr bool is_variadic = false;

  template<typename Y, typename X>
    requires tincup::invocable_c<assign_ftor, Y&, const X&>
  constexpr auto operator()(Y& y, const X& x) const
    noexcept(tincup::nothrow_invocable_c<assign_ftor, Y&, const X&>)
    -> tincup::invocable_t<assign_ftor, Y&, const X&> {
    return tincup::tag_invoke_cpo(*this, y, x);
  }
} assign;

} // namespace rvf


#pragma once

#include "core/real_vector.hpp"
#include "core/cmath/sqrt.hpp"

namespace rvf {

	
inline constexpr struct l2norm_ftor final : tincup::cpo_base<l2norm_ftor> {
  TINCUP_CPO_TAG("l2norm")
  inline static constexpr bool is_variadic = false;
  template<typename V>
  requires tincup::invocable_c<l2norm_ftor, V>
  constexpr auto operator()(V x) const
  noexcept(tincup::nothrow_invocable_c<l2norm_ftor, V>) 
  -> tincup::invocable_t<l2norm_ftor, V> {
    return tincup::tag_invoke_cpo(*this, x);
  }
} l2norm;

template<typename V>
concept l2norm_invocable_c = tincup::invocable_c<l2norm_ftor, V>;

template<typename V>
concept l2norm_nothrow_invocable_c = tincup::nothrow_invocable_c<l2norm_ftor, V>;

template<typename V>
using l2norm_return_t = tincup::invocable_t<l2norm_ftor, V>;

template<typename V>
using l2norm_traits = tincup::cpo_traits<l2norm_ftor, V>;


// Avoid circular dependency with real_vector_c during ADL checks by constraining
// only on the operations actually used (inner_product and sqrt) rather than the
// composite vector concept.
template<typename V>
  requires (
    // Must be able to compute inner_product(x, x)
    tincup::invocable_c<inner_product_ftor, const V&, const V&> &&
    // The result of inner_product must be a valid input to sqrt
    tincup::invocable_c<sqrt_ftor, tincup::invocable_t<inner_product_ftor, const V&, const V&>>
  )
constexpr auto tag_invoke( l2norm_ftor, const V& x ) {
  auto dot = inner_product(x, x);
  return ::rvf::sqrt(dot);
}

} // namespace rvf 

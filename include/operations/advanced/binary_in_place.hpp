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

inline constexpr struct binary_in_place_ftor final : tincup::cpo_base<binary_in_place_ftor> {
  TINCUP_CPO_TAG("binary_in_place")
  inline static constexpr bool is_variadic = false;
  // Re-expose base operator() so failures route to diagnostics
  using tincup::cpo_base<binary_in_place_ftor>::operator();
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename F, typename V>
    requires tincup::invocable_c<binary_in_place_ftor, V&, F, const V>
  constexpr auto operator()(V& x, F&& func, const V y) const
    noexcept(tincup::nothrow_invocable_c<binary_in_place_ftor, V&, F, const V>) 
    -> tincup::invocable_t<binary_in_place_ftor, V&, F, const V> {
    return tag_invoke(*this, x, std::forward<F>(func), y);
  }
} binary_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename F, typename V>
concept binary_in_place_invocable_c = tincup::invocable_c<binary_in_place_ftor, V&, F, const V>;

template<typename F, typename V>
concept binary_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<binary_in_place_ftor, V&, F, const V>;


template<typename F, typename V>
using binary_in_place_return_t = tincup::invocable_t<binary_in_place_ftor, V&, F, const V>;

template<typename F, typename V>
using binary_in_place_traits = tincup::cpo_traits<binary_in_place_ftor, V&, F, const V>;

}
// Usage: tincup::is_invocable_v<binary_in_place_ftor, V&, F, const V>


// External generator-provided argument trait specialization for binary_in_place
// NOTE: Replace YOUR_NAMESPACE with the actual namespace containing your CPO
// (e.g., rvf::binary_in_place_ftor, my_lib::binary_in_place_ftor, etc.)
namespace tincup {
  template<typename F, typename V>
  struct cpo_arg_traits<rvf::binary_in_place_ftor, V&, F&&, const V> {
    static constexpr bool available = true;
    // Fixed (non-pack) argument count
    static constexpr std::size_t fixed_arity = 3;

    // Helpers to build repeated masks for parameter packs
    static constexpr arity_type repeat_mask(std::size_t offset, std::size_t count) {
      arity_type m = arity_type{0};
      for (std::size_t i = 0; i < count; ++i) m |= (arity_type{1} << (offset + i));
      return m;
    }

    // Values mask
    static constexpr arity_type values_mask = []{
      arity_type m = arity_type{0};
      // Fixed positions
      m |= (arity_type{1} << 2);
      // Pack positions
      return m;
    }();

    // Pointers mask
    static constexpr arity_type pointers_mask = []{
      arity_type m = arity_type{0};
      return m;
    }();

    // Lvalue refs mask
    static constexpr arity_type lvalue_refs_mask = []{
      arity_type m = arity_type{0};
m |= (arity_type{1} << 0);      return m;
    }();

    // Rvalue refs mask (non-forwarding)
    static constexpr arity_type rvalue_refs_mask = []{
      arity_type m = arity_type{0};
m |= (arity_type{1} << 1);      return m;
    }();

    // Forwarding refs mask
    static constexpr arity_type forwarding_refs_mask = []{
      arity_type m = arity_type{0};
m |= (arity_type{1} << 1);      return m;
    }();

    // Lvalue const refs mask
    static constexpr arity_type lvalue_const_refs_mask = []{
      arity_type m = arity_type{0};
      return m;
    }();

    // Const-qualified mask (applies to values, refs, or pointers where declared const)
    static constexpr arity_type const_qualified_mask = []{
      arity_type m = arity_type{0};
m |= (arity_type{1} << 2);      return m;
    }();
  };

} // namespace tincup

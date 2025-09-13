/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"

namespace rvf {
	
inline constexpr struct variadic_in_place_ftor final : tincup::cpo_base<variadic_in_place_ftor> {
  TINCUP_CPO_TAG("variadic_in_place")
  inline static constexpr bool is_variadic = true;
  // Re-expose base operator() so failures route to diagnostics
  using tincup::cpo_base<variadic_in_place_ftor>::operator();
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename F, typename V, typename... Vs>
    requires tincup::invocable_c<variadic_in_place_ftor, V&, F, const Vs&...>
  constexpr auto operator()(V& x, F&& func, const Vs&... args) const
    noexcept(tincup::nothrow_invocable_c<variadic_in_place_ftor, V&, F, const Vs&...>) 
    -> tincup::invocable_t<variadic_in_place_ftor, V&, F, const Vs&...> {
    return tag_invoke(*this, x, std::forward<F>(func), args...);
  }
} variadic_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename F, typename V, typename... Vs>
concept variadic_in_place_invocable_c = tincup::invocable_c<variadic_in_place_ftor, V&, F, const Vs&...>;

template<typename F, typename V, typename... Vs>
concept variadic_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<variadic_in_place_ftor, V&, F, const Vs&...>;


template<typename F, typename V, typename... Vs>
using variadic_in_place_return_t = tincup::invocable_t<variadic_in_place_ftor, V&, F, const Vs&...>;

template<typename F, typename V, typename... Vs>
using variadic_in_place_traits = tincup::cpo_traits<variadic_in_place_ftor, V&, F, const Vs&...>;

// Usage: tincup::is_invocable_v<variadic_in_place_ftor, V&, F, const Vs&...>


} // namespace rvf
 
namespace tincup {
  template<typename F, typename V, typename... Vs>
  struct cpo_arg_traits<rvf::variadic_in_place_ftor, V&, F, const Vs&...> {
    static constexpr bool available = true;
    // Fixed (non-pack) argument count
    static constexpr std::size_t fixed_arity = 2;

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
      // Pack positions
        // For packs, category is determined by the declared form of the pack
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
m |= (arity_type{1} << 0);      m |= repeat_mask(fixed_arity, sizeof...(Vs));
      return m;
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
      m |= repeat_mask(fixed_arity, sizeof...(Vs));
      return m;
    }();

    // Const-qualified mask (applies to values, refs, or pointers where declared const)
    static constexpr arity_type const_qualified_mask = []{
      arity_type m = arity_type{0};
      m |= repeat_mask(fixed_arity, sizeof...(Vs));
      return m;
    }();
  };
}


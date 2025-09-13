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


inline constexpr struct axpy_in_place_ftor final : tincup::cpo_base<axpy_in_place_ftor> {
  TINCUP_CPO_TAG("axpy_in_place")
  inline static constexpr bool is_variadic = false;
  // Re-expose base operator() so failures route to diagnostics
  using tincup::cpo_base<axpy_in_place_ftor>::operator();
    // Typed operator() overload - positive case only (generic)
  // Negative cases handled by tagged fallback in cpo_base
  template<typename S, typename V>
    requires tincup::invocable_c<axpy_in_place_ftor, V&, S, const V&>
  constexpr auto operator()(V& y, S alpha, const V& x) const
    noexcept(tincup::nothrow_invocable_c<axpy_in_place_ftor, V&, S, const V&>) 
    -> tincup::invocable_t<axpy_in_place_ftor, V&, S, const V&> {
    return tag_invoke(*this, y, alpha, x);
  }
} axpy_in_place;

// Note: operator() methods are provided by cpo_base

// CPO-specific concepts and type aliases for convenient usage
template<typename S, typename V>
concept axpy_in_place_invocable_c = tincup::invocable_c<axpy_in_place_ftor, V&, S, const V&>;

template<typename S, typename V>
concept axpy_in_place_nothrow_invocable_c = tincup::nothrow_invocable_c<axpy_in_place_ftor, V&, S, const V&>;

template<typename S, typename V>
using axpy_in_place_return_t = tincup::invocable_t<axpy_in_place_ftor, V&, S, const V&>;

template<typename S, typename V>
using axpy_in_place_traits = tincup::cpo_traits<axpy_in_place_ftor, V&, S, const V&>;

// Usage: tincup::is_invocable_v<axpy_in_place_ftor, V&, S, const V&>
/**
 * @brief Default tag_invoke overload for axpy_in_place
 * @tag_invoke_impl
 * Provides a default implementation that performs y = alpha * x + y
 * by cloning x, scaling the clone by alpha, then adding to y.
 *
 * @param y Target vector (modified in-place)
 * @param alpha Scalar multiplier for vector x
 * @param x Source vector (not modified)
 * @return void
 */
template<typename S, typename V>
constexpr auto tag_invoke( axpy_in_place_ftor, V& y, S alpha, const V& x ) -> void {
  auto x_clone = clone(x); auto& alphax = deref_if_needed(x_clone);
  scale_in_place(alphax, alpha);
  add_in_place(y, alphax);
}

} // namespace rvf

namespace tincup {
  template<typename S, typename V>
  struct cpo_arg_traits<rvf::axpy_in_place_ftor, V&, S, const V&> {
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
      m |= (arity_type{1} << 1);
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
      m |= (arity_type{1} << 0);m |= (arity_type{1} << 2);      
      return m;
    }();

    // Rvalue refs mask (non-forwarding)
    static constexpr arity_type rvalue_refs_mask = []{
      arity_type m = arity_type{0};
      return m;
    }();

    // Forwarding refs mask
    static constexpr arity_type forwarding_refs_mask = []{
      arity_type m = arity_type{0};
      return m;
    }();

    // Lvalue const refs mask
    static constexpr arity_type lvalue_const_refs_mask = []{
      arity_type m = arity_type{0};
      m |= (arity_type{1} << 2);      
      return m;
    }();

    // Const-qualified mask (applies to values, refs, or pointers where declared const)
    static constexpr arity_type const_qualified_mask = []{
      arity_type m = arity_type{0};
      m |= (arity_type{1} << 2);      
      return m;
    }();
  };
} // namespace tincup



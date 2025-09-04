/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <utility> 

// Core concepts and type aliases for vector-like types in RVF.
//
// - real_scalar_c<T>: scalar type concept (defaults to std::floating_point, can be overridden)
// - real_vector_c<V>: vector concept defined in terms of RVF CPOs and their return types
// - inner_product_return_t<V>, dimension_return_t<V>, clone_return_t<V>: convenience aliases
//
// Include the CPO declarations used by the concept definitions below.
#include "operations/core/add_in_place.hpp"
#include "operations/core/clone.hpp"
#include "operations/core/inner_product.hpp"
#include "operations/core/dimension.hpp"
#include "operations/core/scale_in_place.hpp"
#include "operations/core/deref_if_needed.hpp"

#ifdef CUSTOM_REAL_SCALAR
#include "real_scalar.hpp"
#else
template<class T>
struct is_real_scalar : std::is_floating_point<T> {};

template<typename T>
concept real_scalar_c = std::floating_point<T>;
#endif

namespace rvf {

template<typename Cp, typename... Args>
concept returns_real_scalar_c = tincup::cpo_c<Cp> && Cp::template valid_return_type<is_real_scalar,Args...>;   

// Type aliases for convenience
template<typename V>
using clone_return_t = tincup::invocable_t<clone_ftor, const V&>;

template<typename V>
using inner_product_return_t = tincup::invocable_t<inner_product_ftor, const V&, const V&>;

template<typename V>
using dimension_return_t = tincup::invocable_t<dimension_ftor, const V&>;

// Concepts for return type constraints
template<typename V>
concept returns_clone_c = tincup::invocable_c<clone_ftor, const V&>;

template<typename V>
concept real_vector_c = 
  // Argument Type Requirements
  tincup::invocable_c<add_in_place_ftor,V&, const V&> &&
  tincup::invocable_c<clone_ftor,const V&> &&
  tincup::invocable_c<dimension_ftor,const V&> &&
  tincup::invocable_c<inner_product_ftor,const V&,const V&> &&
  tincup::invocable_c<scale_in_place_ftor,V&,inner_product_return_t<V>> &&
  // Return Type Requirements
  tincup::returns_void_c<add_in_place_ftor, V&, const V&> &&
  returns_clone_c<V> &&
  tincup::returns_integral_c<dimension_ftor, const V&> &&
  returns_real_scalar_c<inner_product_ftor, const V&, const V&> &&
  tincup::returns_void_c<scale_in_place_ftor, V&, inner_product_return_t<V>>;

// Additional constraint to ensure clone result is a real vector
template<typename T>
  requires real_vector_c<T>
struct clone_compatibility_check {
  using clone_result = deref_t<clone_return_t<T>>;
  static_assert(real_vector_c<clone_result>, 
    "Clone result must satisfy real_vector_c after dereferencing");
};

} // namespace rvf

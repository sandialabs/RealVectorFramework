/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "operations/advanced/binary_in_place.hpp"
#include "operations/advanced/fill.hpp"
#include "operations/advanced/layer_norm.hpp"
#include "operations/advanced/matvec.hpp"
#include "operations/advanced/relu.hpp"
#include "operations/advanced/softmax.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include "operations/advanced/variadic_in_place.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <ranges>
#include <cmath>
#include <type_traits>

// tag_invoke overloads for any type that satisfies std::ranges::range and is copy constructible
// Define these in namespace rvf so ADL finds them via the CPO argument.

namespace rvf {

// add_in_place
template<std::ranges::range R>
void tag_invoke(add_in_place_ftor, R& y, const R& x) {
  std::ranges::transform(y, x, std::ranges::begin(y), std::plus<>{});
}
	
// clone
template<std::ranges::range R>
requires std::copy_constructible<R>
auto tag_invoke(clone_ftor, const R& x) {
  return R(x);
}

// dimension
template<std::ranges::range R>
auto tag_invoke(dimension_ftor, const R& r) {
  return std::ranges::size(r);
}

// inner_product
template<std::ranges::range R>
auto tag_invoke(inner_product_ftor, const R& x, const R& y) {
  using value_type = std::ranges::range_value_t<R>;
  return std::inner_product(std::ranges::cbegin(x), 
                            std::ranges::cend(x), 
                            std::ranges::begin(y), 
                            static_cast<value_type>(0));
}

// scale_in_place
template<std::ranges::range R>
void tag_invoke(scale_in_place_ftor, R& y, std::ranges::range_value_t<R> alpha) {
  std::ranges::for_each(y, [alpha](auto& ye){ ye *= alpha; });
}

// fill
template<std::ranges::range R>
void tag_invoke(fill_ftor, R& y, std::ranges::range_value_t<R> alpha) {
  std::ranges::for_each(y, [alpha](auto& ye){ ye = alpha; });
}

// unary_in_place
template<typename F, typename T>
concept unary_in_place_invocable = 
  std::convertible_to<std::invoke_result_t<F,T>,T>;

template<std::ranges::range R, typename F>
requires unary_in_place_invocable<F, std::ranges::range_value_t<R>>
void tag_invoke( unary_in_place_ftor, R& y, F&& func ) {
  std::ranges::for_each(y, [func = std::forward<F>(func)](auto& ye) mutable { 
    ye = func(ye); 
  });
}

// binary_in_place
template<typename F, typename T>
concept binary_in_place_invocable = 
  std::convertible_to<std::invoke_result_t<F,T,T>,T>;

template<std::ranges::range R, typename F>
requires binary_in_place_invocable<F, std::ranges::range_value_t<R>>
void tag_invoke( binary_in_place_ftor, R& x, F&& func, const R& y ) {
  std::ranges::transform(x, y, std::ranges::begin(x), std::forward<F>(func));
}

// variadic_in_place
template<typename F, typename T, typename...Args>
concept variadic_in_place_invocable =
  std::convertible_to<std::invoke_result_t<F,T,Args...>,T> &&
  (std::is_same_v<T,Args> && ...);

template<class OutputIt, class F, class FirstInput, class...RestInput>
OutputIt variadic_transform( OutputIt     out_begin,
                             OutputIt     out_end,
                             F&&          func,
                             FirstInput   first,
                             RestInput... rest ) { 
  while(out_begin != out_end) {
    *out_begin++ = std::forward<F>(func)(it_inc(first), it_inc(rest)...);
  }
  return out_begin;
}

template<std::ranges::range R, typename F, typename...Args>
requires variadic_in_place_invocable<F, std::ranges::range_value_t<R>, 
	                                std::ranges::range_value_t<Args>...>
void tag_invoke( variadic_in_place_ftor, R& x, F&& func, const Args&... args ) {
  variadic_transform(
    std::ranges::begin(x),
    std::ranges::end(x),
    std::forward<F>(func),
    std::ranges::begin(args)...
  );    
}

// relu
template<std::ranges::range R>
void tag_invoke(relu_ftor, R& target) {
  std::ranges::for_each(target, [](auto& x) { 
    using T = std::decay_t<decltype(x)>;
    x = std::max(static_cast<T>(0), x); 
  });
}

// softmax
template<std::ranges::range R>
void tag_invoke(softmax_ftor, R& target) {
  using value_type = std::ranges::range_value_t<R>;
  
  // Find maximum for numerical stability
  auto max_it = std::ranges::max_element(target);
  if (max_it == std::ranges::end(target)) return;  // Empty range
  
  value_type max_val = *max_it;
  
  // Subtract max and exponentiate using rvf::exp
  std::ranges::for_each(target, [max_val](auto& x) {
    x = rvf::exp(x - max_val);
  });

  // Sum via inner_product with a ones vector
  R ones = target;
  rvf::fill(ones, static_cast<value_type>(1));
  value_type sum = rvf::inner_product(ones, target);
  
  // Normalize
  if (sum > static_cast<value_type>(0)) {
    std::ranges::for_each(target, [sum](auto& x) { x /= sum; });
  }
}

// layer_norm (single argument - default eps)
template<std::ranges::range R>
void tag_invoke(layer_norm_ftor, R& target) {
  using value_type = std::ranges::range_value_t<R>;
  constexpr value_type default_eps = static_cast<value_type>(1e-5);
  tag_invoke(layer_norm_ftor{}, target, default_eps);
}

// layer_norm (with eps parameter)
template<std::ranges::range R, typename Scalar>
requires std::convertible_to<Scalar, std::ranges::range_value_t<R>>
void tag_invoke(layer_norm_ftor, R& target, Scalar eps) {
  using value_type = std::ranges::range_value_t<R>;
  const auto n = std::ranges::size(target);
  if (n == 0) return;  // Empty range
  
  // Compute mean
  value_type mean = static_cast<value_type>(0);
  for (const auto& x : target) {
    mean += x;
  }
  mean /= static_cast<value_type>(n);
  
  // Subtract mean
  std::ranges::for_each(target, [mean](auto& x) { x -= mean; });
  
  // Compute variance
  value_type var = static_cast<value_type>(0);
  for (const auto& x : target) {
    var += x * x;
  }
  var /= static_cast<value_type>(n);
  
  // Normalize
  value_type norm_factor = static_cast<value_type>(1) / 
                          std::sqrt(var + static_cast<value_type>(eps));
  std::ranges::for_each(target, [norm_factor](auto& x) { x *= norm_factor; });
}

// matvec (matrix-vector multiplication)
template<std::ranges::range V, std::ranges::range M, std::ranges::range X>
requires std::ranges::range<std::ranges::range_value_t<M>>  // M is range of ranges
void tag_invoke(matvec_ftor, V& y, const M& A, const X& x) {
  auto y_it = std::ranges::begin(y);
  auto y_end = std::ranges::end(y);
  
  for (const auto& row : A) {
    if (y_it == y_end) break;  // Prevent out-of-bounds access
    
    // Compute dot product of row with x
    using value_type = std::ranges::range_value_t<V>;
    *y_it = std::inner_product(
      std::ranges::cbegin(row), std::ranges::cend(row),
      std::ranges::cbegin(x),
      static_cast<value_type>(0)
    );
    ++y_it;
  }
}

} // namespace rvf

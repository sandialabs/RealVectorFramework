/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include "operations/advanced/binary_in_place.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <ranges>

// tag_invoke overloads for any type that satisfies std::ranges::range and is copy constructible
// Define these in namespace rvf so ADL finds them via the CPO argument.

namespace rvf {

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

// add_in_place
template<std::ranges::range R>
void tag_invoke(add_in_place_ftor, R& y, const R& x) {
  std::ranges::transform(y, x, std::ranges::begin(y), std::plus<>{});
}

// scale_in_place
template<std::ranges::range R>
void tag_invoke(scale_in_place_ftor, R& y, std::ranges::range_value_t<R> alpha) {
  std::ranges::for_each(y, [alpha](auto& ye){ ye *= alpha; });
}

// unary_in_place
template<typename F, typename T>
concept unary_in_place_invocable = std::convertible_to<
  std::invoke_result_t<F, T>,
  T
>;

template<std::ranges::range R, typename F>
  requires unary_in_place_invocable<F, std::ranges::range_value_t<R>>
void tag_invoke(unary_in_place_ftor, R& y, F&& func) {
  std::ranges::for_each(y, [func = std::forward<F>(func)](auto& ye) mutable { 
    ye = func(ye); 
  });
}

// binary_in_place
template<typename F, typename T>
concept binary_in_place_invocable = std::convertible_to<
  std::invoke_result_t<F, T, T>,
  T
>;

template<std::ranges::range R, typename F>
  requires binary_in_place_invocable<F, std::ranges::range_value_t<R>>
void tag_invoke(binary_in_place_ftor, R& y, const R& x, F&& func) {
  std::ranges::transform(y, x, std::ranges::begin(y), std::forward<F>(func));
}

// clone
template<std::ranges::range R>
  requires std::copy_constructible<R>
auto tag_invoke(clone_ftor, const R& x) {
  return R(x);
}

} // namespace rvf

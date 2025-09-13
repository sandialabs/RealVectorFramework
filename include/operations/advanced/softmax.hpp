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

/**
 * @brief Apply softmax function to vector (in-place)
 * @cpo
 * Applies the softmax function to the target vector, modifying it in place.
 * Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
 * 
 * The implementation uses the numerically stable version:
 * Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)) for all j)
 * 
 * This is commonly used as the final activation function in classification
 * neural networks, as it converts logits to probability distributions.
 *
 * @cpo_example
 * @code
 * std::vector<double> logits = {1.0, 2.0, 3.0};
 * softmax(logits);  // logits becomes probability distribution summing to 1.0
 * @endcode
 */
inline constexpr struct softmax_ftor final : tincup::cpo_base<softmax_ftor> {
  TINCUP_CPO_TAG("softmax")
  inline static constexpr bool is_variadic = false;
  
  template<typename V>
    requires tincup::invocable_c<softmax_ftor, V&>
  constexpr auto operator()(V& target) const
    noexcept(tincup::nothrow_invocable_c<softmax_ftor, V&>) 
    -> tincup::invocable_t<softmax_ftor, V&> {
    return tincup::tag_invoke_cpo(*this, target);
  }
} softmax;

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept softmax_invocable_c = tincup::invocable_c<softmax_ftor, V&>;

template<typename V>
concept softmax_nothrow_invocable_c = tincup::nothrow_invocable_c<softmax_ftor, V&>;

template<typename V>
using softmax_return_t = tincup::invocable_t<softmax_ftor, V&>;

template<typename V>
using softmax_traits = tincup::cpo_traits<softmax_ftor, V&>;

} // namespace rvf
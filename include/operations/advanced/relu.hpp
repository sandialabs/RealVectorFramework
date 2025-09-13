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
 * @brief Apply ReLU activation function elementwise (in-place)
 * @cpo
 * Applies the Rectified Linear Unit (ReLU) function to each element of the
 * target vector, modifying the vector in place. ReLU(x) = max(0, x).
 *
 * This is a fundamental activation function used in neural networks and
 * machine learning applications.
 *
 * @cpo_example
 * @code
 * std::vector<double> vec = {-2.0, -1.0, 0.0, 1.0, 2.0};
 * relu(vec);  // vec becomes {0.0, 0.0, 0.0, 1.0, 2.0}
 * @endcode
 */
inline constexpr struct relu_ftor final : tincup::cpo_base<relu_ftor> {
  TINCUP_CPO_TAG("relu")
  inline static constexpr bool is_variadic = false;
  
  template<typename V>
    requires tincup::invocable_c<relu_ftor, V&>
  constexpr auto operator()(V& target) const
    noexcept(tincup::nothrow_invocable_c<relu_ftor, V&>) 
    -> tincup::invocable_t<relu_ftor, V&> {
    return tincup::tag_invoke_cpo(*this, target);
  }
} relu;

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept relu_invocable_c = tincup::invocable_c<relu_ftor, V&>;

template<typename V>
concept relu_nothrow_invocable_c = tincup::nothrow_invocable_c<relu_ftor, V&>;

template<typename V>
using relu_return_t = tincup::invocable_t<relu_ftor, V&>;

template<typename V>
using relu_traits = tincup::cpo_traits<relu_ftor, V&>;

} // namespace rvf
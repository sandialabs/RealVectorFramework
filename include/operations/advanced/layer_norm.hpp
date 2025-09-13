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
 * @brief Apply layer normalization to vector (in-place)
 * @cpo
 * Applies layer normalization to the target vector, modifying it in place.
 * Layer normalization normalizes the activations of each layer to have 
 * zero mean and unit variance:
 * 
 * LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)
 * 
 * where mean(x) and var(x) are computed over all elements of the vector,
 * and eps is a small constant for numerical stability.
 * 
 * This is commonly used in transformer architectures and other deep learning
 * models to stabilize training and improve convergence.
 *
 * @param target The vector to normalize
 * @param eps Small constant for numerical stability (default: 1e-5)
 *
 * @cpo_example
 * @code
 * std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
 * layer_norm(vec);        // Normalize with default eps=1e-5
 * layer_norm(vec, 1e-6);  // Normalize with custom eps
 * @endcode
 */
inline constexpr struct layer_norm_ftor final : tincup::cpo_base<layer_norm_ftor> {
  TINCUP_CPO_TAG("layer_norm")
  inline static constexpr bool is_variadic = false;
  
  // Single argument version (uses default eps)
  template<typename V>
    requires tincup::invocable_c<layer_norm_ftor, V&>
  constexpr auto operator()(V& target) const
    noexcept(tincup::nothrow_invocable_c<layer_norm_ftor, V&>) 
    -> tincup::invocable_t<layer_norm_ftor, V&> {
    return tincup::tag_invoke_cpo(*this, target);
  }
  
  // Two argument version (with custom eps)
  template<typename V, typename Scalar>
    requires tincup::invocable_c<layer_norm_ftor, V&, Scalar>
  constexpr auto operator()(V& target, Scalar eps) const
    noexcept(tincup::nothrow_invocable_c<layer_norm_ftor, V&, Scalar>) 
    -> tincup::invocable_t<layer_norm_ftor, V&, Scalar> {
    return tincup::tag_invoke_cpo(*this, target, eps);
  }
} layer_norm;

// CPO-specific concepts and type aliases for convenient usage
template<typename V>
concept layer_norm_invocable_c = tincup::invocable_c<layer_norm_ftor, V&>;

template<typename V, typename Scalar>
concept layer_norm_eps_invocable_c = tincup::invocable_c<layer_norm_ftor, V&, Scalar>;

template<typename V>
concept layer_norm_nothrow_invocable_c = tincup::nothrow_invocable_c<layer_norm_ftor, V&>;

template<typename V, typename Scalar>
concept layer_norm_eps_nothrow_invocable_c = tincup::nothrow_invocable_c<layer_norm_ftor, V&, Scalar>;

template<typename V>
using layer_norm_return_t = tincup::invocable_t<layer_norm_ftor, V&>;

template<typename V, typename Scalar>
using layer_norm_eps_return_t = tincup::invocable_t<layer_norm_ftor, V&, Scalar>;

template<typename V>
using layer_norm_traits = tincup::cpo_traits<layer_norm_ftor, V&>;

template<typename V, typename Scalar>
using layer_norm_eps_traits = tincup::cpo_traits<layer_norm_ftor, V&, Scalar>;

} // namespace rvf
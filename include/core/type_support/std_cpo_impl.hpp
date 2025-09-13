/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

// Illustration of using tincup::cpo_impl specializations (formatter-style)
// for standard containers. This header is optional and not included by
// default; include it from a TU to activate the trait-based implementations.

#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#include <tincup/tincup.hpp>
#include "operations/advanced/fill.hpp"
#include "core/real_vector.hpp"

// -----------------------------------------------------------------------------
// Trait specializations in namespace tincup
// -----------------------------------------------------------------------------
namespace tincup {

// add_in_place for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::add_in_place_ftor, std::vector<T, Alloc>> {
  static void call(std::vector<T, Alloc>& y, const std::vector<T, Alloc>& x) {
    std::transform(y.begin(), y.end(), x.begin(), y.begin(), std::plus<>{});
  }
};

// scale_in_place for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::scale_in_place_ftor, std::vector<T, Alloc>> {
  static void call(std::vector<T, Alloc>& y, T alpha) {
    std::for_each(y.begin(), y.end(), [alpha](T& v){ v *= alpha; });
  }
};

// inner_product for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::inner_product_ftor, std::vector<T, Alloc>> {
  static T call(const std::vector<T, Alloc>& x, const std::vector<T, Alloc>& y) {
    return std::inner_product(x.begin(), x.end(), y.begin(), T{});
  }
};

// axpy_in_place for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::axpy_in_place_ftor, std::vector<T, Alloc>> {
  static void call(std::vector<T, Alloc>& y, T alpha, const std::vector<T, Alloc>& x) {
    std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                   [alpha](const T& x_val, const T& y_val) {
                     return alpha * x_val + y_val;
                   });
  }
};

// dimension for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::dimension_ftor, std::vector<T, Alloc>> {
  static std::size_t call(const std::vector<T, Alloc>& x) {
    return x.size();
  }
};

// clone for std::vector<T, Alloc>
template<typename T, typename Alloc>
struct cpo_impl<rvf::clone_ftor, std::vector<T, Alloc>> {
  static auto call(const std::vector<T, Alloc>& x) {
    return std::vector<T, Alloc>(x);
  }
};

} // namespace tincup


// fill for std::vector<T, Alloc>
namespace tincup {
template<typename T, typename Alloc>
struct cpo_impl<rvf::fill_ftor, std::vector<T, Alloc>> {
  static void call(std::vector<T, Alloc>& v, const T& value) {
    std::fill(v.begin(), v.end(), value);
  }
};

} // namespace tincup


// -----------------------------------------------------------------------------
// ADL-visible shims in namespace rvf that forward to the trait impls
// -----------------------------------------------------------------------------
namespace rvf {

template<typename T, typename Alloc>
constexpr auto tag_invoke( add_in_place_ftor, 
		           std::vector<T, Alloc>& y, 
			   const std::vector<T,Alloc>& x ) {
  return tincup::cpo_impl<add_in_place_ftor, std::vector<T, Alloc>>::call(y, x); 
}
            
template<typename T, typename Alloc>
constexpr auto tag_invoke( scale_in_place_ftor, 
		           std::vector<T, Alloc>& y, 
			   T alpha ) {
  return tincup::cpo_impl<scale_in_place_ftor, std::vector<T, Alloc>>::call(y, alpha);
}

template<typename T, typename Alloc>
constexpr auto tag_invoke( inner_product_ftor, 
	                   const std::vector<T, Alloc>& x, 
			   const std::vector<T,Alloc>& y ) {
  return tincup::cpo_impl<inner_product_ftor, std::vector<T, Alloc>>::call(x,y); 
}
            
template<typename T, typename Alloc>
constexpr auto tag_invoke(dimension_ftor, const std::vector<T, Alloc>& x) {
 return tincup::cpo_impl<dimension_ftor, std::vector<T, Alloc>>::call(x); 
}

template<typename T, typename Alloc>
constexpr auto tag_invoke(clone_ftor, const std::vector<T, Alloc>& x ) {
  return tincup::cpo_impl<clone_ftor, std::vector<T, Alloc>>::call(x); 
}

template<typename T, typename Alloc>
constexpr auto tag_invoke(axpy_in_place_ftor, 
                         std::vector<T, Alloc>& y, 
                         T alpha,
                         const std::vector<T, Alloc>& x) {
  return tincup::cpo_impl<axpy_in_place_ftor, std::vector<T, Alloc>>::call(y, alpha, x);
}

} // namespace rvf

namespace rvf {

template<typename T, typename Alloc>
constexpr auto tag_invoke(fill_ftor, std::vector<T, Alloc>& v, const T& value) {
  return tincup::cpo_impl<fill_ftor, std::vector<T, Alloc>>::call(v, value);
}

} // namespace rvf


// Usage:
//  - Include this header in a TU to route std::vector operations through
//    tincup::cpo_impl specializations instead of the generic ranges path.
//  - This is for illustration; keep the generic ranges-based tag_invoke
//    implementations as the default behavior.

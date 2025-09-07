#pragma once

#include <cmath>
#include "core/cmath/sqrt.hpp"

namespace rvf {

inline float tag_invoke( sqrt_ftor, float x ) {
  return std::sqrt(x);
}

inline double tag_invoke( sqrt_ftor, double x ) {
  return std::sqrt(x);
}

inline long double tag_invoke( sqrt_ftor, long double x ) {
  return std::sqrt(x);
}

} // namespace rvf 

#pragma once

#include <cmath>
#include "core/cmath/fmin.hpp"

namespace rvf {

inline float tag_invoke( fmin_ftor, float x, float y ) {
  return std::fmin(x,y);
}

inline double tag_invoke( fmin_ftor, double x, double y ) {
  return std::fmin(x,y);
}

inline long double tag_invoke( fmin_ftor, long double x, long double y ) {
  return std::fmin(x,y);
}

} // namespace rvf 

#pragma once

#include <cmath>
#include "core/cmath/remainder.hpp"

namespace rvf {

inline float tag_invoke( remainder_ftor, float x, float y ) {
  return std::remainder(x,y);
}

inline double tag_invoke( remainder_ftor, double x, double y ) {
  return std::remainder(x,y);
}

inline long double tag_invoke( remainder_ftor, long double x, long double y ) {
  return std::remainder(x,y);
}

} // namespace rvf

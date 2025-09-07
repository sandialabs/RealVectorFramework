#pragma once

#include <cmath>
#include "core/cmath/fmax.hpp"

namespace rvf {

inline float tag_invoke( fmax_ftor, float x, float y ) {
  return std::fmax(x,y);
}

inline double tag_invoke( fmax_ftor, double x, double y ) {
  return std::fmax(x,y);
}

inline long double tag_invoke( fmax_ftor, long double x, long double y ) {
  return std::fmax(x,y);
}

} // namespace rvf 

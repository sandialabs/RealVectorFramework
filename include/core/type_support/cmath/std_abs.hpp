#pragma once 

#include <cmath>
#include "core/cmath/abs.hpp"

namespace rvf {

inline float tag_invoke( abs_ftor, float x ) {
  return std::fabs(x);
}

inline double tag_invoke( abs_ftor, double x ) {
  return std::fabs(x);
}

inline long double tag_invoke( abs_ftor, long double x ) {
  return std::fabs(x);
}

} // namespace rvf


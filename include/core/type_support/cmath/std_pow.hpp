#pragma once

#include <cmath>
#include "core/cmath/pow.hpp"

namespace rvf {

inline float tag_invoke( pow_ftor, float base, float exp ) {
  return std::pow(base,exp);
}  

inline double tag_invoke( pow_ftor, double base, double exp ) {
  return std::pow(base,exp);
}  

inline long double tag_invoke( pow_ftor, long double base, long double exp ) {
  return std::pow(base,exp);
}  

} // namespace rvf 

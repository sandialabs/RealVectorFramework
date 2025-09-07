#pragma once

#include <cmath>
#include "core/cmath/fmod.hpp"

namespace rvf {

inline float tag_invoke( fmod_ftor, float x, float y ) {
  return std::fmod(x,y);
}	

inline double tag_invoke( fmod_ftor, double x, double y ) {
  return std::fmod(x,y);
}	

inline long double tag_invoke( fmod_ftor, long double x, long double y ) {
  return std::fmod(x,y);
}	

} // namespace rvf 



/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

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

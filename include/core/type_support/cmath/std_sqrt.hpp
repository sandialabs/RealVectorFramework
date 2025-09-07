/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

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

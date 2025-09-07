/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <cmath>
#include "core/cmath/fma.hpp"

namespace rvf {

inline float tag_invoke( fma_ftor, float x, float y, float z ) {
  return std::fma(x,y,z);
}

inline double tag_invoke( fma_ftor, double x, double y, double z ) {
  return std::fma(x,y,z);
}

inline long double tag_invoke( fma_ftor, long double x, long double y, long double z ) {
  return std::fma(x,y,z);
}

} // namespace rvf

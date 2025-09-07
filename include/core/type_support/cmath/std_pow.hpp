/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

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

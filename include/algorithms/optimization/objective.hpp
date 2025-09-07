/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"

namespace rvf {

template<typename Obj, typename Vec>
concept objective_c = requires( const Obj& obj, const Vec& x, Vec& g ) {
  { obj.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
  { obj.gradient(g,x) } -> std::same_as<void>; 
};

template<typename Obj, typename Vec>
concept objective_hess_vec_c = requires( const Obj& obj, const Vec& x, const Vec& v, Vec& hv ) { 
  { obj.hessVec(hv,v,x) } -> std::same_as<void>; 
};	

template<typename Obj, typename Vec>
concept objective_preconditioner_c = requires( const Obj& obj, const Vec& x, const Vec& v, Vec& Pv ) { 
  { obj.precond(Pv,v,x) } -> std::same_as<void>; 
};	


} // nanespace rvf

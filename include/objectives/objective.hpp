#pragma once

#include "real_vector.hpp"

namespace rvf {

// Objective function concept
template<typename F, typename Vec>
concept objective_value_c = requires( const F& f, const Vec& x ) {
  { f.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
};

template<typename F, typename Vec>
concept objective_gradient_c = requires( const F& f, const Vec& x, Vec& g ) { 
  { f.gradient(g,x) } -> std::same_as<void>; 
};

template<typename F, typename Vec>
concept objective_hess_vec_c = requires( const F& f, const Vec& x, const Vec& v, Vec& hv ) { 
  { f.hessVec(hv,v,x) } -> std::same_as<void>; 
};	

} // nanespace rvf

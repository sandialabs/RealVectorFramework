/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "algorithms/optimization/objective.hpp"

namespace rvf {

template<typename P>
using x0_t = decltype(std::declval<P>().x0);

template<typename P>
using objective_t = decltype(std::declval<P>().objective);

template<typename P>
concept unconstrained_problem_c = real_vector_c<x0_t<P>> &&
                                  objective_c<objective_t<P>,x0_t<P>>;

} // namespace rvf

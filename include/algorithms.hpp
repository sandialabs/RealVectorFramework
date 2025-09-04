/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

// Include all algorithms

// Linear algebra algorithms
#include "algorithms/linear_algebra/conjugate_gradient.hpp"
#include "algorithms/linear_algebra/sherman_morrison.hpp"

// Optimization algorithms
#include "algorithms/optimization/trust_region/truncated_cg.hpp"
#include "algorithms/optimization/line_search/gradient_descent_bounds.hpp"

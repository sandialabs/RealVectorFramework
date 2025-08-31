/**
TInCuP - A library for generating and validating C++ customization point objects that use `tag_invoke`

Copyright (c) National Technology & Engineering Solutions of Sandia, 
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

// Include all operations
#include "operations/add_in_place.hpp"
#include "operations/clone.hpp"
#include "operations/inner_product.hpp"
#include "operations/dimension.hpp"
#include "operations/scale_in_place.hpp"
#include "operations/axpy_in_place.hpp"
#include "operations/unary_in_place.hpp"
#include "operations/binary_in_place.hpp"
#include "operations/variadic_in_place.hpp"
#include "operations/deref_if_needed.hpp"
#include "operations/std_ranges_support.hpp"
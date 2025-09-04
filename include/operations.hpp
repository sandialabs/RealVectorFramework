/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

// Aggregate header for all RVF operations/CPOs
// Include this to access the full set of vector operations.
// Generic implementations for ranges live in core/type_support/std_ranges_support.hpp.
// Backend-specific specializations may be provided via tincup::cpo_impl.

// Core operations (required for real_vector_c)
#include "operations/core/add_in_place.hpp"
#include "operations/core/clone.hpp"
#include "operations/core/inner_product.hpp"
#include "operations/core/dimension.hpp"
#include "operations/core/scale_in_place.hpp"
#include "operations/core/deref_if_needed.hpp"

// Advanced operations (convenience/composite operations)
#include "operations/advanced/axpy_in_place.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include "operations/advanced/binary_in_place.hpp"
#include "operations/advanced/variadic_in_place.hpp"
#include "operations/advanced/self_map.hpp"

// Memory management operations
#include "operations/memory/memory_arena.hpp"
#include "operations/memory/arena_integration.hpp"
#include "operations/memory/arena_observers.hpp"

// Type support
#include "core/type_support/std_ranges_support.hpp"

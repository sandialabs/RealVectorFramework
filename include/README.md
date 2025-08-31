# Real Vector Framework Include Structure

This directory contains the Real Vector Framework headers, organized into a clear modular structure:

## Directory Structure

```
include/
├── rvf.hpp                    # Main header - includes everything
├── real_vector.hpp            # Core concepts and basic CPOs
├── operations.hpp             # All operations - convenience header
├── algorithms.hpp             # All algorithms - convenience header
├── operations/                # Vector operations and CPOs
│   ├── add_in_place.hpp
│   ├── clone.hpp
│   ├── inner_product.hpp
│   ├── dimension.hpp
│   ├── scale_in_place.hpp
│   ├── axpy_in_place.hpp
│   ├── unary_in_place.hpp
│   ├── binary_in_place.hpp
│   ├── variadic_in_place.hpp
│   ├── deref_if_needed.hpp
│   └── std_ranges_support.hpp
└── algorithms/                # Higher-level algorithms
    ├── conjugate_gradient.hpp
    └── sherman_morrison.hpp
```

## Usage

### Include Everything
```cpp
#include "rvf.hpp"  // Includes all RVF functionality
```

### Include Selectively
```cpp
#include "real_vector.hpp"    // Just core concepts and basic CPOs
#include "operations.hpp"     // All operations
#include "algorithms.hpp"     // All algorithms
```

### Include Specific Components
```cpp
#include "operations/axpy_in_place.hpp"          // Specific operation
#include "algorithms/conjugate_gradient.hpp"    // Specific algorithm
```

## Operations vs Algorithms

- **Operations** (`operations/`): Basic vector operations and CPOs that work element-wise or vector-wise
- **Algorithms** (`algorithms/`): Higher-level algorithms that use the operations to solve mathematical problems

All files use the new refactored TInCuP library for consistent CPO implementation.
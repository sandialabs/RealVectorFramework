# RVF API Reference Documentation

This directory contains comprehensive API reference documentation for the Real Vector Framework (RVF). The documentation is organized by functionality and provides detailed information for developers using and extending RVF.

## Documentation Structure

### 📋 [Main API Reference](API_REFERENCE.md)
Comprehensive overview of all RVF APIs with examples and usage patterns. Start here for a complete overview.

### 🧩 [Core Concepts](CONCEPTS.md)
Detailed reference for RVF's concept system:
- `real_scalar_c<T>` - Scalar type concept
- `real_vector_c<V>` - Vector type concept
- Type aliases and utilities
- Concept composition and customization

### ⚙️ [Operations (CPOs)](OPERATIONS.md)
Complete reference for Customization Point Objects:
- **Core Operations**: `clone`, `dimension`, `inner_product`, `add_in_place`, `scale_in_place`
- **Advanced Operations**: `axpy_in_place`, `unary_in_place`, `l2norm`, neural network ops
- **Specialization Guide**: How to implement CPOs for custom types
- **Performance Optimization**: SIMD, caching, specialization techniques

### 🧮 [Algorithms](ALGORITHMS.md)
Comprehensive algorithm reference:
- **Linear Algebra**: Conjugate gradient, Sherman-Morrison formula
- **Optimization**: Trust region methods (TruncatedCG), line search methods
- **Objective Functions**: Framework for optimization problems
- **Algorithm Composition**: Combining algorithms for complex solvers

### 🏗️ [Memory Management](MEMORY.md)
Advanced memory management systems:
- **Memory Arenas**: High-performance allocation pools
- **Arena Integration**: CPO-based arena support
- **Performance Analysis**: Benchmarks and optimization strategies
- **Best Practices**: Sizing, error handling, monitoring

## Quick Reference

### Basic Usage
```cpp
#include "rvf.hpp"
#include "core/type_support/std_ranges_support.hpp"

using namespace rvf;
using Vector = std::vector<double>;

Vector x = {1.0, 2.0, 3.0};
auto y_clone = clone(x);
auto& y = deref_if_needed(y_clone);

scale_in_place(y, 2.0);           // y *= 2.0
axpy_in_place(y, 1.5, x);         // y += 1.5 * x
double dot = inner_product(x, y);  // x · y
```

### Trust Region Optimization
```cpp
auto objective = make_zakharov_objective(k_vector);
TruncatedCG<decltype(objective), Vector> solver(objective, x);

Vector step(n, 0.0);
auto result = solver.solve(x, step, trust_radius, params);
```

### High-Performance Computing
```cpp
MemoryArena arena(64 * 1024 * 1024);  // 64MB pool

{
    auto scope = arena.create_scope();

    // All operations use fast arena allocation
    auto temp1 = arena_clone(x, arena);
    auto temp2 = arena_clone(y, arena);

    // ... computations ...

} // Automatic cleanup
```

## API Categories

| Category | Core Components | Purpose |
|----------|----------------|---------|
| **Concepts** | `real_scalar_c`, `real_vector_c` | Type safety and generic programming |
| **Core Ops** | `clone`, `inner_product`, `add_in_place` | Fundamental vector operations |
| **Advanced Ops** | `axpy_in_place`, `l2norm`, neural nets | Convenience and specialized operations |
| **Linear Algebra** | `conjugate_gradient`, `sherman_morrison` | System solving and matrix operations |
| **Optimization** | `TruncatedCG`, `gradient_descent_bounds` | Nonlinear optimization algorithms |
| **Memory** | `MemoryArena`, arena integration | High-performance memory management |

## Implementation Patterns

### CPO Specialization
Two primary approaches for extending RVF to custom types:

#### 1. ADL `tag_invoke` (Recommended for types you control)
```cpp
namespace my_namespace {
struct MyVector { /* ... */ };

auto tag_invoke(rvf::clone_ftor, const MyVector& v) -> MyVector {
    return MyVector(v);
}
}
```

#### 2. `tincup::cpo_impl` Specialization (For third-party types)
```cpp
namespace tincup {
template<typename T>
struct cpo_impl<rvf::clone_ftor, ThirdPartyVector<T>> {
    static auto call(const ThirdPartyVector<T>& v) -> ThirdPartyVector<T> {
        return ThirdPartyVector<T>(v);
    }
};
}
```

### Algorithm Patterns
```cpp
template<real_vector_c Vector>
auto my_algorithm(Vector& x) -> result_type {
    // 1. Clone for temporaries
    auto temp = clone(x);
    auto& temp_ref = deref_if_needed(temp);

    // 2. Use CPO operations
    scale_in_place(temp_ref, 2.0);
    add_in_place(x, temp_ref);

    // 3. Return computed results
    return inner_product(x, temp_ref);
}
```

## Performance Characteristics

### CPO Overhead
- **Compile-time**: Zero overhead with proper specialization
- **Runtime**: Identical to hand-written code after optimization
- **Template instantiation**: Minimal due to concept constraints

### Memory Arena Benefits
- **Small vectors (< 1KB)**: ~0.5x performance (stack allocation better)
- **Medium vectors (1-100KB)**: **10-30x** faster allocation
- **Large vectors (> 100KB)**: **100-5000x** faster allocation
- **Algorithm patterns**: **10-50x** improvement in iterative algorithms

### Benchmark Summary
Based on included benchmarks (Samsung Galaxy Book Pro):

| Operation | Standard | Arena | Improvement |
|-----------|----------|-------|-------------|
| Clone 10K elements | 1993 ns | 65.7 ns | **30x faster** |
| Clone 1M elements | 358504 ns | 65.0 ns | **5516x faster** |
| Algorithm pattern (100K) | 1.09ms | 1.9μs | **573x faster** |

## Integration Guide

### CMake Integration
```cmake
add_subdirectory(RealVectorFramework)
target_link_libraries(your_target PRIVATE rvf::rvf)
```

### Header Organization
```cpp
// Include everything
#include "rvf.hpp"

// Or selectively
#include "core/real_vector.hpp"    // Concepts only
#include "operations.hpp"          // All operations
#include "algorithms.hpp"          // All algorithms
```

### Compiler Requirements
- **C++20** compatible compiler (GCC 10+, Clang 12+, MSVC 2022+)
- **CMake 3.14+** for building examples and tests
- **TInCuP** library (automatically fetched via CMake)

## Troubleshooting

### Common Issues

#### Concept Failures
```cpp
// Error: Type doesn't satisfy real_vector_c
static_assert(real_vector_c<MyVector>);  // Add this check

// Solution: Implement required CPOs
auto tag_invoke(rvf::clone_ftor, const MyVector& v) { /* ... */ }
```

#### Wrapper Type Issues
```cpp
// Error: Temporary wrapper passed to constrained function
function_requiring_real_vector_c(clone(x));  // May fail

// Solution: Use recommended idiom
auto cl = clone(x);
auto& ref = deref_if_needed(cl);
function_requiring_real_vector_c(ref);  // Always works
```

#### Performance Issues
```cpp
// Slow: Repeated allocations
for (int i = 0; i < 1000; ++i) {
    auto temp = clone(large_vector);  // malloc/free each iteration
}

// Fast: Use arena
MemoryArena arena(estimated_size);
auto scope = arena.create_scope();
for (int i = 0; i < 1000; ++i) {
    auto temp = arena_clone(large_vector, arena);  // Arena allocation
}
```

## Contributing

### Documentation Standards
- Use **Markdown** for all documentation
- Include **code examples** for all APIs
- Provide **mathematical definitions** where applicable
- Add **performance considerations** for algorithms
- Include **common pitfalls** and solutions

### Code Examples
All code examples should:
- Compile with current RVF headers
- Follow established naming conventions
- Include necessary includes
- Demonstrate best practices

---

## See Also

- [Main README](../../README.md) - Project overview and quick start
- [CPO Integration Guide](../CPO_INTEGRATION_GUIDE.md) - Detailed CPO implementation guide
- [Benchmark Results](../BENCHMARKS.md) - Performance analysis
- [Examples Directory](../../examples/) - Working code examples
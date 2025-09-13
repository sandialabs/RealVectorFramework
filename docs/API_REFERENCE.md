# RVF API Reference

The Real Vector Framework (RVF) provides a comprehensive C++20 API for generic vector operations and numerical algorithms. This reference documents all public APIs organized by functionality.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Core Operations (CPOs)](#core-operations)
3. [Advanced Operations](#advanced-operations)
4. [Memory Management](#memory-management)
5. [Mathematical Functions](#mathematical-functions)
6. [Linear Algebra Algorithms](#linear-algebra-algorithms)
7. [Optimization Algorithms](#optimization-algorithms)
8. [Type Support](#type-support)

---

## Core Concepts

### `real_scalar_c<T>`
```cpp
template<typename T>
concept real_scalar_c = std::floating_point<T>;
```
**Purpose**: Defines what types are considered real scalars in RVF.
**Default**: Any C++ floating-point type (`float`, `double`, `long double`).
**Customization**: Can be overridden by defining `CUSTOM_REAL_SCALAR` and providing custom `real_scalar.hpp`.

### `real_vector_c<V>`
```cpp
template<typename V>
concept real_vector_c = /* ... */;
```
**Purpose**: The fundamental concept defining vector-like types in RVF.
**Requirements**: A type `V` satisfies `real_vector_c` if it supports:
- `add_in_place(v, x)` - In-place vector addition
- `clone(v)` - Vector copying/cloning
- `dimension(v)` - Size/dimension query
- `inner_product(v, x)` - Dot product computation
- `scale_in_place(v, α)` - In-place scalar multiplication

**Return type requirements**: All operations must return appropriate types (void for in-place ops, scalars for reductions, etc.).

### Type Aliases
```cpp
// Convenience aliases for common return types
template<typename V> using clone_return_t = /* deduced clone return type */;
template<typename V> using inner_product_return_t = /* deduced scalar type */;
template<typename V> using dimension_return_t = /* deduced size type */;
```

---

## Core Operations

Core operations are the fundamental building blocks required by `real_vector_c`. All are Customization Point Objects (CPOs) implemented using TInCuP.

### `clone`
```cpp
auto clone(const V& v) -> /* implementation-defined */;
```
**Purpose**: Creates a copy of vector `v`.
**Returns**: May return either a value (e.g., `std::vector<double>`) or a wrapper (e.g., `std::unique_ptr<std::vector<double>>`).
**Usage Pattern**:
```cpp
auto cl = rvf::clone(x);
auto& xr = rvf::deref_if_needed(cl);
// Use xr as regular vector, cl owns storage
```

### `dimension`
```cpp
auto dimension(const V& v) -> std::integral auto;
```
**Purpose**: Returns the size/dimension of vector `v`.
**Returns**: Integral type representing vector size.
**Example**:
```cpp
std::vector<double> v{1, 2, 3};
assert(dimension(v) == 3);
```

### `inner_product`
```cpp
auto inner_product(const V& x, const V& y) -> real_scalar_c auto;
```
**Purpose**: Computes dot product `x · y`.
**Returns**: Real scalar type.
**Mathematical**: `∑ᵢ xᵢ * yᵢ`
**Example**:
```cpp
std::vector<double> x{1, 2, 3};
std::vector<double> y{4, 5, 6};
double dot = inner_product(x, y); // 32.0
```

### `add_in_place`
```cpp
void add_in_place(V& y, const V& x);
```
**Purpose**: In-place vector addition `y ← y + x`.
**Mathematical**: `yᵢ ← yᵢ + xᵢ` for all i.
**Example**:
```cpp
std::vector<double> y{1, 2, 3};
std::vector<double> x{4, 5, 6};
add_in_place(y, x); // y becomes {5, 7, 9}
```

### `scale_in_place`
```cpp
void scale_in_place(V& v, const Scalar& α);
```
**Purpose**: In-place scalar multiplication `v ← α * v`.
**Mathematical**: `vᵢ ← α * vᵢ` for all i.
**Example**:
```cpp
std::vector<double> v{1, 2, 3};
scale_in_place(v, 2.0); // v becomes {2, 4, 6}
```

### `deref_if_needed`
```cpp
auto& deref_if_needed(auto&& wrapper);
```
**Purpose**: Dereferences wrapper types, passes through regular types unchanged.
**Use case**: Obtaining references from `clone()` results that may be wrappers.
**Example**:
```cpp
auto cl = clone(v);        // Might return unique_ptr<vector<double>>
auto& vref = deref_if_needed(cl); // Always returns vector<double>&
```

---

## Advanced Operations

Advanced operations provide convenience and composite functionality built on core operations.

### `axpy_in_place`
```cpp
void axpy_in_place(V& y, const Scalar& α, const V& x);
```
**Purpose**: Scaled vector addition `y ← y + α * x` (BLAS AXPY operation).
**Mathematical**: `yᵢ ← yᵢ + α * xᵢ` for all i.
**Example**:
```cpp
std::vector<double> y{1, 2, 3};
std::vector<double> x{4, 5, 6};
axpy_in_place(y, 2.0, x); // y becomes {9, 12, 15}
```

### `unary_in_place`
```cpp
void unary_in_place(V& v, UnaryFunction f);
```
**Purpose**: Applies unary function element-wise in-place.
**Mathematical**: `vᵢ ← f(vᵢ)` for all i.
**Example**:
```cpp
std::vector<double> v{1, 4, 9};
unary_in_place(v, [](double x) { return std::sqrt(x); }); // v becomes {1, 2, 3}
```

### `binary_in_place`
```cpp
void binary_in_place(V& z, const V& x, const V& y, BinaryFunction f);
```
**Purpose**: Applies binary function element-wise: `z ← f(x, y)`.
**Mathematical**: `zᵢ ← f(xᵢ, yᵢ)` for all i.
**Example**:
```cpp
std::vector<double> x{1, 2, 3}, y{4, 5, 6}, z(3);
binary_in_place(z, x, y, std::plus<>{}); // z becomes {5, 7, 9}
```

### `variadic_in_place`
```cpp
void variadic_in_place(V& result, VariadicFunction f, const Args&... args);
```
**Purpose**: Applies variadic function element-wise with multiple vector arguments.
**Mathematical**: `resultᵢ ← f(args[0][i], args[1][i], ...)` for all i.

### `l2norm`
```cpp
auto l2norm(const V& v) -> real_scalar_c auto;
```
**Purpose**: Computes L2 (Euclidean) norm of vector.
**Mathematical**: `||v||₂ = √(∑ᵢ vᵢ²)`
**Implementation**: `std::sqrt(inner_product(v, v))`

### Neural Network Operations

#### `relu`
```cpp
void relu(V& v);
```
**Purpose**: Applies ReLU activation in-place: `v ← max(0, v)`.

#### `softmax`
```cpp
void softmax(V& v);
```
**Purpose**: Applies softmax activation in-place with numerical stability.

#### `layer_norm`
```cpp
void layer_norm(V& v);
```
**Purpose**: Applies layer normalization in-place.

---

## Memory Management

RVF provides advanced memory arena systems for high-performance computing scenarios.

### `MemoryArena`
```cpp
class MemoryArena {
public:
    explicit MemoryArena(size_t capacity);
    auto create_scope() -> ScopedArena;
    // ... implementation details
};
```
**Purpose**: Memory pool for efficient temporary allocations.
**Benefits**: 10-100x faster allocation for small vectors, reduced fragmentation.

### Arena Integration CPOs

#### `arena_integration`
```cpp
// CPO for arena-aware allocation
```
**Purpose**: Enables automatic arena allocation for vector operations.

#### `arena_observers`
```cpp
// CPO for memory usage tracking
```
**Purpose**: Provides hooks for monitoring arena memory usage.

---

## Mathematical Functions

RVF extends standard mathematical functions to work with CPOs and custom types.

### Available Functions
- `abs(x)` - Absolute value
- `sqrt(x)` - Square root
- `exp(x)` - Exponential
- `pow(x, y)` - Power function
- `fmax(x, y)`, `fmin(x, y)` - Min/max with NaN handling
- `fmod(x, y)` - Floating-point remainder
- `remainder(x, y)` - IEEE remainder
- `fma(x, y, z)` - Fused multiply-add

All functions work with both scalar types and can be extended via CPO specialization.

---

## Linear Algebra Algorithms

### `conjugate_gradient`
```cpp
template<typename LinearOperator, typename Vector>
auto conjugate_gradient(
    const LinearOperator& A,
    const Vector& b,
    Vector& x,
    double tolerance = 1e-8,
    int max_iterations = -1
) -> CGResult;
```
**Purpose**: Solves `A*x = b` using conjugate gradient iteration for symmetric positive definite systems.
**Parameters**:
- `A`: Linear operator (function or object with `operator()`)
- `b`: Right-hand side vector
- `x`: Initial guess (modified in-place)
- `tolerance`: Convergence tolerance
- `max_iterations`: Maximum iterations (-1 for automatic)

### `sherman_morrison_solve`
```cpp
template<typename LinearOperator, typename Vector>
void sherman_morrison_solve(
    const LinearOperator& A_inv_action,
    const Vector& u, const Vector& v,
    const Vector& b, Vector& x
);
```
**Purpose**: Solves `(A + u*v^T)*x = b` efficiently when `A^{-1}` action is available.
**Mathematical**: Uses Sherman-Morrison formula for rank-1 updates.
**Complexity**: O(n) when A⁻¹ action is O(n).

---

## Optimization Algorithms

### Trust Region Methods

#### `TruncatedCG`
```cpp
template<typename Objective, typename Vector>
class TruncatedCG {
public:
    TruncatedCG(const Objective& obj, const Vector& template_vec);

    auto solve(const Vector& x_current, Vector& step,
               double trust_radius, const Params& params) -> Result;

    struct Params {
        double abs_tol = 1e-8;
        double rel_tol = 1e-2;
        int max_iter = -1;
    };

    struct Result {
        TerminationStatus status;
        int iter;
        double step_norm;
        double pred_reduction;
    };
};
```
**Purpose**: Steihaug-Toint truncated conjugate gradient for trust region subproblems.
**Problem**: Minimize `g^T*s + (1/2)*s^T*H*s` subject to `||s|| ≤ δ`.
**Features**: Handles negative curvature, boundary detection.

### Line Search Methods

#### `gradient_descent_bounds`
```cpp
template<typename Objective, typename Vector>
void gradient_descent_bounds(
    const Objective& objective,
    const bound_constraints<Vector>& bounds,
    Vector& x,
    double grad_tolerance = 1e-6,
    double initial_step_size = 1.0,
    int max_iterations = 1000
);
```
**Purpose**: Projected gradient descent with bound constraints.
**Problem**: Minimize `f(x)` subject to `lower ≤ x ≤ upper`.
**Features**: Backtracking line search, projection onto feasible region.

### Objective Function Framework

#### Core Concepts
```cpp
template<typename F, typename Vec>
concept objective_value_c = requires(const F& f, const Vec& x) {
    { f.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
};

template<typename F, typename Vec>
concept objective_gradient_c = requires(const F& f, const Vec& x, Vec& g) {
    { f.gradient(g, x) } -> std::same_as<void>;
};

template<typename F, typename Vec>
concept objective_hess_vec_c = requires(const F& f, const Vec& x, const Vec& v, Vec& hv) {
    { f.hessVec(hv, v, x) } -> std::same_as<void>;
};
```

#### Test Problems

##### Zakharov Function
```cpp
auto make_zakharov_objective(const Vector& k);
```
**Mathematical**: `f(x) = x^T*x + (1/4)*(k^T*x)^2 + (1/16)*(k^T*x)^4`
**Purpose**: Standard test problem for optimization algorithms.

---

## Type Support

### Standard Library Integration
```cpp
#include "core/type_support/std_ranges_support.hpp"
```
**Provides**: Generic CPO implementations for `std::ranges::range` types.
**Coverage**: `std::vector`, `std::array`, spans, views, etc.

### Specialized Implementations
```cpp
#include "core/type_support/std_cpo_impl.hpp"
```
**Provides**: Optimized `tincup::cpo_impl` specializations for `std::vector`.
**Usage**: Include to enable trait-based routing for better performance.

### Custom Type Integration
To integrate custom vector types, implement required CPOs:

```cpp
// Option 1: ADL tag_invoke (in your namespace)
auto tag_invoke(rvf::clone_ftor, const MyVector& v) -> MyVector {
    return MyVector(v);
}

// Option 2: tincup::cpo_impl specialization
namespace tincup {
template<typename T>
struct cpo_impl<rvf::clone_ftor, MyVector<T>> {
    static auto call(const MyVector<T>& v) -> MyVector<T> {
        return MyVector<T>(v);
    }
};
}
```

---

## Integration Examples

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
#include "rvf.hpp"

const size_t n = 5;
Vector x(n, 3.0), k(n);
std::iota(k.begin(), k.end(), 1.0);

auto objective = make_zakharov_objective(k);
TruncatedCG<decltype(objective), Vector> solver(objective, x);

Vector step(n, 0.0);
auto result = solver.solve(x, step, 1.0, {});
```

### Custom Backend Integration
```cpp
// GPU vector example
class CudaVector { /* ... */ };

// Implement CPOs for GPU operations
auto tag_invoke(rvf::inner_product_ftor, const CudaVector& x, const CudaVector& y) -> double {
    return cuda_dot_product(x, y);  // CUDA kernel
}

// Now CudaVector works with all RVF algorithms
conjugate_gradient(gpu_matrix, gpu_b, gpu_x, 1e-8);
```
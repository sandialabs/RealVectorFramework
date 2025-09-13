# RVF Operations (CPOs) Reference

This document provides comprehensive reference for all RVF Customization Point Objects (CPOs). Operations are organized by category and complexity level.

## Overview

RVF operations are implemented as Customization Point Objects (CPOs) using the TInCuP library. This design allows for:
- Type-safe generic programming
- Efficient specialization for custom types
- Consistent interfaces across all vector types

---

## Core Operations

Core operations are required for `real_vector_c` compliance and form the foundation of all RVF algorithms.

### `clone`

```cpp
TINCUP_DEFINE_CPO(clone_ftor, clone)
auto clone(const V& v) -> /* implementation-defined */;
```

**Purpose**: Creates a copy of vector `v`.

**Parameters**:
- `v`: Vector to clone (const reference)

**Returns**: Implementation-defined type that may be:
- Value type (e.g., `std::vector<double>`)
- Wrapper type (e.g., `std::unique_ptr<std::vector<double>>`)

**Thread Safety**: Implementation-dependent

**Example**:
```cpp
std::vector<double> x{1, 2, 3};
auto x_copy = clone(x);
auto& x_ref = deref_if_needed(x_copy);  // Recommended pattern
```

**Specialization Example**:
```cpp
// Custom vector type
struct MyVector { /* ... */ };

// ADL specialization
auto tag_invoke(rvf::clone_ftor, const MyVector& v) -> MyVector {
    return MyVector(v);
}

// Or trait specialization
namespace tincup {
template<>
struct cpo_impl<rvf::clone_ftor, MyVector> {
    static auto call(const MyVector& v) -> MyVector {
        return MyVector(v);
    }
};
}
```

---

### `dimension`

```cpp
TINCUP_DEFINE_CPO(dimension_ftor, dimension)
auto dimension(const V& v) -> std::integral auto;
```

**Purpose**: Returns the size/dimension of vector `v`.

**Parameters**:
- `v`: Vector to query (const reference)

**Returns**: Integral type representing vector size

**Complexity**: O(1) for most implementations

**Example**:
```cpp
std::vector<double> v{1, 2, 3, 4, 5};
assert(dimension(v) == 5);

std::array<float, 10> arr{};
assert(dimension(arr) == 10);
```

**Requirements for Custom Types**:
```cpp
static_assert(std::integral<decltype(dimension(my_vector))>);
```

---

### `inner_product`

```cpp
TINCUP_DEFINE_CPO(inner_product_ftor, inner_product)
auto inner_product(const V& x, const V& y) -> real_scalar_c auto;
```

**Purpose**: Computes dot product `x · y = ∑ᵢ xᵢ * yᵢ`.

**Parameters**:
- `x`: First vector (const reference)
- `y`: Second vector (const reference)

**Returns**: Real scalar type (same as vector element type)

**Preconditions**:
- `dimension(x) == dimension(y)`
- Both vectors must have the same element type

**Mathematical Definition**: `x · y = ∑ᵢ₌₀ⁿ⁻¹ xᵢ * yᵢ`

**Complexity**: O(n) where n = dimension(x)

**Example**:
```cpp
std::vector<double> x{1, 2, 3};
std::vector<double> y{4, 5, 6};
double result = inner_product(x, y);  // 1*4 + 2*5 + 3*6 = 32
```

**Numerical Considerations**:
- Implementation should be numerically stable
- Consider Kahan summation for high precision requirements
- GPU implementations may use different reduction strategies

---

### `add_in_place`

```cpp
TINCUP_DEFINE_CPO(add_in_place_ftor, add_in_place)
void add_in_place(V& y, const V& x);
```

**Purpose**: In-place vector addition `y ← y + x`.

**Parameters**:
- `y`: Target vector (mutable reference)
- `x`: Source vector (const reference)

**Returns**: `void`

**Preconditions**: `dimension(y) == dimension(x)`

**Mathematical Definition**: `yᵢ ← yᵢ + xᵢ` for all i ∈ [0, n)

**Complexity**: O(n)

**Example**:
```cpp
std::vector<double> y{1, 2, 3};
std::vector<double> x{4, 5, 6};
add_in_place(y, x);
// y is now {5, 7, 9}
```

**SIMD Optimization**:
```cpp
// Example specialization for SIMD
namespace tincup {
template<>
struct cpo_impl<rvf::add_in_place_ftor, std::vector<double>> {
    static void call(std::vector<double>& y, const std::vector<double>& x) {
        // Use vectorized addition
        #pragma omp simd
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] += x[i];
        }
    }
};
}
```

---

### `scale_in_place`

```cpp
TINCUP_DEFINE_CPO(scale_in_place_ftor, scale_in_place)
void scale_in_place(V& v, const Scalar& alpha);
```

**Purpose**: In-place scalar multiplication `v ← α * v`.

**Parameters**:
- `v`: Vector to scale (mutable reference)
- `alpha`: Scalar multiplier

**Returns**: `void`

**Mathematical Definition**: `vᵢ ← α * vᵢ` for all i ∈ [0, n)

**Complexity**: O(n)

**Example**:
```cpp
std::vector<double> v{1, 2, 3};
scale_in_place(v, 2.5);
// v is now {2.5, 5.0, 7.5}
```

**Special Cases**:
```cpp
scale_in_place(v, 0.0);   // Zeros the vector
scale_in_place(v, 1.0);   // No-op (implementations may optimize)
scale_in_place(v, -1.0);  // Negates the vector
```

---

### `deref_if_needed`

```cpp
TINCUP_DEFINE_CPO(deref_if_needed_ftor, deref_if_needed)
auto& deref_if_needed(auto&& wrapper);
```

**Purpose**: Dereferences pointer-like wrapper types, passes through value types unchanged.

**Parameters**:
- `wrapper`: Object that might need dereferencing

**Returns**: Reference to the underlying object

**Use Cases**:
- Handling `clone()` results that may be smart pointers
- Working with optional-like wrapper types
- Memory arena integration

**Example**:
```cpp
// Case 1: clone returns value
std::vector<double> v{1, 2, 3};
auto copy1 = clone(v);  // Returns std::vector<double>
auto& ref1 = deref_if_needed(copy1);  // Returns std::vector<double>&

// Case 2: clone returns wrapper
auto copy2 = arena_clone(v);  // Returns std::unique_ptr<std::vector<double>>
auto& ref2 = deref_if_needed(copy2);  // Returns std::vector<double>&
```

**Implementation Pattern**:
```cpp
// For pointer-like types
template<typename T>
auto tag_invoke(rvf::deref_if_needed_ftor, std::unique_ptr<T>& ptr) -> T& {
    return *ptr;
}

// For value types (identity)
template<typename T>
auto tag_invoke(rvf::deref_if_needed_ftor, T& value) -> T& {
    return value;
}
```

---

## Advanced Operations

Advanced operations provide convenience and composite functionality built on core operations.

### `axpy_in_place`

```cpp
TINCUP_DEFINE_CPO(axpy_in_place_ftor, axpy_in_place)
void axpy_in_place(V& y, const Scalar& alpha, const V& x);
```

**Purpose**: Scaled vector addition `y ← y + α * x` (BLAS AXPY).

**Parameters**:
- `y`: Target vector (mutable reference)
- `alpha`: Scalar multiplier
- `x`: Source vector (const reference)

**Returns**: `void`

**Mathematical Definition**: `yᵢ ← yᵢ + α * xᵢ` for all i ∈ [0, n)

**Default Implementation**:
```cpp
// Generic implementation using core operations
auto temp = clone(x);
auto& temp_ref = deref_if_needed(temp);
scale_in_place(temp_ref, alpha);
add_in_place(y, temp_ref);
```

**Optimized Implementation**:
```cpp
// Direct implementation avoiding temporary
for (size_t i = 0; i < dimension(y); ++i) {
    y[i] += alpha * x[i];
}
```

**Example**:
```cpp
std::vector<double> y{1, 2, 3};
std::vector<double> x{4, 5, 6};
axpy_in_place(y, 2.0, x);
// y is now {9, 12, 15} = {1, 2, 3} + 2.0 * {4, 5, 6}
```

---

### `unary_in_place`

```cpp
TINCUP_DEFINE_CPO(unary_in_place_ftor, unary_in_place)
void unary_in_place(V& v, UnaryFunction f);
```

**Purpose**: Applies unary function element-wise in-place.

**Parameters**:
- `v`: Vector to transform (mutable reference)
- `f`: Unary function or callable object

**Returns**: `void`

**Mathematical Definition**: `vᵢ ← f(vᵢ)` for all i ∈ [0, n)

**Function Requirements**: `f` must be callable as `f(element_type)`

**Example**:
```cpp
std::vector<double> v{1, 4, 9, 16};
unary_in_place(v, [](double x) { return std::sqrt(x); });
// v is now {1, 2, 3, 4}

unary_in_place(v, std::negate<>{});  // Negate all elements
```

**Performance Note**: Implementations may vectorize simple functions.

---

### `binary_in_place`

```cpp
TINCUP_DEFINE_CPO(binary_in_place_ftor, binary_in_place)
void binary_in_place(V& result, const V& x, const V& y, BinaryFunction f);
```

**Purpose**: Applies binary function element-wise: `result ← f(x, y)`.

**Parameters**:
- `result`: Output vector (mutable reference)
- `x`: First input vector (const reference)
- `y`: Second input vector (const reference)
- `f`: Binary function or callable object

**Returns**: `void`

**Mathematical Definition**: `resultᵢ ← f(xᵢ, yᵢ)` for all i ∈ [0, n)

**Preconditions**: All vectors must have the same dimension

**Example**:
```cpp
std::vector<double> x{1, 2, 3};
std::vector<double> y{4, 5, 6};
std::vector<double> result(3);

binary_in_place(result, x, y, std::plus<>{});
// result is now {5, 7, 9}

binary_in_place(result, x, y, std::multiplies<>{});
// result is now {4, 10, 18}
```

---

### `variadic_in_place`

```cpp
TINCUP_DEFINE_CPO(variadic_in_place_ftor, variadic_in_place)
void variadic_in_place(V& result, VariadicFunction f, const Args&... args);
```

**Purpose**: Applies variadic function element-wise with multiple vector arguments.

**Parameters**:
- `result`: Output vector (mutable reference)
- `f`: Variadic function
- `args`: Input vectors (const references)

**Mathematical Definition**: `resultᵢ ← f(args[0][i], args[1][i], ...)` for all i

**Example**:
```cpp
std::vector<double> a{1, 2, 3};
std::vector<double> b{4, 5, 6};
std::vector<double> c{7, 8, 9};
std::vector<double> result(3);

auto weighted_sum = [](double x, double y, double z) {
    return 0.5*x + 0.3*y + 0.2*z;
};

variadic_in_place(result, weighted_sum, a, b, c);
// result[i] = 0.5*a[i] + 0.3*b[i] + 0.2*c[i]
```

---

### `l2norm`

```cpp
TINCUP_DEFINE_CPO(l2norm_ftor, l2norm)
auto l2norm(const V& v) -> real_scalar_c auto;
```

**Purpose**: Computes L2 (Euclidean) norm of vector.

**Parameters**:
- `v`: Input vector (const reference)

**Returns**: Real scalar representing the L2 norm

**Mathematical Definition**: `‖v‖₂ = √(∑ᵢ vᵢ²) = √(v · v)`

**Default Implementation**: `std::sqrt(inner_product(v, v))`

**Numerical Considerations**:
- May overflow for very large vectors
- Consider scaled norm for numerical stability

**Example**:
```cpp
std::vector<double> v{3, 4};
double norm = l2norm(v);  // 5.0
```

---

## Specialized Operations

### Neural Network Operations

#### `relu`
```cpp
TINCUP_DEFINE_CPO(relu_ftor, relu)
void relu(V& v);
```
**Purpose**: Applies ReLU activation: `v ← max(0, v)` element-wise.
**Implementation**: `unary_in_place(v, [](auto x) { return std::max(x, 0); })`

#### `softmax`
```cpp
TINCUP_DEFINE_CPO(softmax_ftor, softmax)
void softmax(V& v);
```
**Purpose**: Applies softmax activation with numerical stability.
**Mathematical**: `vᵢ ← exp(vᵢ - max(v)) / ∑ⱼ exp(vⱼ - max(v))`

#### `layer_norm`
```cpp
TINCUP_DEFINE_CPO(layer_norm_ftor, layer_norm)
void layer_norm(V& v);
```
**Purpose**: Applies layer normalization: `v ← (v - mean(v)) / std(v)`.

### Utility Operations

#### `assign`
```cpp
void assign(V& target, const V& source);
```
**Purpose**: Copies elements from source to target vector.

#### `fill`
```cpp
void fill(V& v, const Scalar& value);
```
**Purpose**: Fills vector with constant value.

#### `matvec`
```cpp
void matvec(V& y, const Matrix& A, const V& x);
```
**Purpose**: Matrix-vector multiplication `y ← A * x`.

---

## CPO Implementation Guide

### Basic CPO Structure
```cpp
namespace rvf {
// 1. Define the CPO type
struct my_operation_ftor {
    template<typename... Args>
    constexpr auto operator()(Args&&... args) const
        -> decltype(tincup::cpo_invoke(*this, std::forward<Args>(args)...))
    {
        return tincup::cpo_invoke(*this, std::forward<Args>(args)...);
    }
};

// 2. Create global CPO instance
inline constexpr my_operation_ftor my_operation{};
}
```

### Specialization Methods

#### Method 1: ADL `tag_invoke`
```cpp
// In your type's namespace
namespace my_namespace {
struct MyVector { /* ... */ };

auto tag_invoke(rvf::clone_ftor, const MyVector& v) -> MyVector {
    return MyVector(v);  // Custom clone implementation
}
}
```

#### Method 2: `tincup::cpo_impl` Specialization
```cpp
namespace tincup {
template<typename T>
struct cpo_impl<rvf::clone_ftor, MyVector<T>> {
    static auto call(const MyVector<T>& v) -> MyVector<T> {
        return MyVector<T>(v);
    }
};
}
```

### Performance Optimization Tips

1. **Vectorization**: Use SIMD-friendly loops
2. **Memory Access**: Optimize cache locality
3. **Branch Prediction**: Minimize conditional operations
4. **Template Specialization**: Provide optimized versions for common types
5. **Compile-Time Evaluation**: Use `constexpr` where possible

### Testing CPO Implementations
```cpp
// Concept compliance test
static_assert(real_vector_c<MyVector>);

// Functionality test
void test_my_vector() {
    MyVector v1{1, 2, 3};
    auto v2 = clone(v1);
    assert(dimension(v1) == dimension(v2));
    assert(inner_product(v1, v2) > 0);
}
```
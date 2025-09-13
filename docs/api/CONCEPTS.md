# RVF Core Concepts Reference

This document provides detailed reference information for RVF's core concepts and type system.

## Overview

RVF uses C++20 concepts to provide type safety and clear interfaces. The concept hierarchy is designed to be both flexible and performant.

## Concept Hierarchy

```
real_scalar_c<T>          // Floating-point scalars
    ↓
real_vector_c<V>          // Vector types supporting core CPOs
    ↓
[Algorithm-specific concepts]
```

---

## `real_scalar_c<T>`

### Definition
```cpp
#ifdef CUSTOM_REAL_SCALAR
#include "real_scalar.hpp"  // User-provided definition
#else
template<typename T>
concept real_scalar_c = std::floating_point<T>;
#endif
```

### Purpose
Defines what types can be used as scalars in vector operations. By default, this includes all standard floating-point types.

### Standard Types
- `float`
- `double`
- `long double`

### Customization
To use custom scalar types (e.g., fixed-point, autodiff), define `CUSTOM_REAL_SCALAR` before including RVF headers and provide `real_scalar.hpp`:

```cpp
// real_scalar.hpp
#pragma once
#include <type_traits>

template<class T>
struct is_real_scalar : std::false_type {};

template<>
struct is_real_scalar<double> : std::true_type {};

template<>
struct is_real_scalar<MyCustomScalar> : std::true_type {};

template<typename T>
concept real_scalar_c = is_real_scalar<T>::value;
```

---

## `real_vector_c<V>`

### Definition
```cpp
template<typename V>
concept real_vector_c =
  // Invocability requirements
  tincup::invocable_c<add_in_place_ftor, V&, const V&> &&
  tincup::invocable_c<clone_ftor, const V&> &&
  tincup::invocable_c<dimension_ftor, const V&> &&
  tincup::invocable_c<inner_product_ftor, const V&, const V&> &&
  tincup::invocable_c<scale_in_place_ftor, V&, inner_product_return_t<V>> &&

  // Return type requirements
  tincup::returns_void_c<add_in_place_ftor, V&, const V&> &&
  returns_clone_c<V> &&
  tincup::returns_integral_c<dimension_ftor, const V&> &&
  returns_real_scalar_c<inner_product_ftor, const V&, const V&> &&
  tincup::returns_void_c<scale_in_place_ftor, V&, inner_product_return_t<V>>;
```

### Purpose
The fundamental concept for vector-like types in RVF. A type satisfying this concept can be used with all RVF algorithms.

### Requirements

#### Required Operations
1. **`add_in_place(v, x)`** - In-place addition
2. **`clone(v)`** - Vector copying
3. **`dimension(v)`** - Size query
4. **`inner_product(v, x)`** - Dot product
5. **`scale_in_place(v, α)`** - In-place scaling

#### Return Type Constraints
- `add_in_place` and `scale_in_place` must return `void`
- `dimension` must return an integral type
- `inner_product` must return a `real_scalar_c` type
- `clone` may return value or wrapper type

### Standard Compliant Types
With appropriate CPO implementations:
- `std::vector<T>` where `real_scalar_c<T>`
- `std::array<T, N>` where `real_scalar_c<T>`
- `std::span<T>` where `real_scalar_c<T>`
- Standard library ranges/views (with generic implementations)

### Clone Compatibility
Types must satisfy additional clone compatibility:
```cpp
template<typename T>
requires real_vector_c<T>
struct clone_compatibility_check {
  using clone_result = deref_t<clone_return_t<T>>;
  static_assert(real_vector_c<clone_result>,
    "Clone result must satisfy real_vector_c after dereferencing");
};
```

---

## Type Aliases

### `clone_return_t<V>`
```cpp
template<typename V>
using clone_return_t = tincup::invocable_t<clone_ftor, const V&>;
```
**Purpose**: Deduces the return type of `clone(v)` for vector type `V`.
**Usage**: Template metaprogramming and concept definitions.

### `inner_product_return_t<V>`
```cpp
template<typename V>
using inner_product_return_t = tincup::invocable_t<inner_product_ftor, const V&, const V&>;
```
**Purpose**: Deduces the scalar type returned by `inner_product(v, v)`.
**Usage**: Ensuring scalar type consistency across operations.

### `dimension_return_t<V>`
```cpp
template<typename V>
using dimension_return_t = tincup::invocable_t<dimension_ftor, const V&>;
```
**Purpose**: Deduces the size type returned by `dimension(v)`.
**Usage**: Size computations and loop bounds.

---

## Utility Concepts

### `returns_clone_c<V>`
```cpp
template<typename V>
concept returns_clone_c = tincup::invocable_c<clone_ftor, const V&>;
```
**Purpose**: Checks if a type can be cloned.

### `returns_real_scalar_c<Cp, Args...>`
```cpp
template<typename Cp, typename... Args>
concept returns_real_scalar_c =
    tincup::cpo_c<Cp> &&
    Cp::template valid_return_type<is_real_scalar, Args...>;
```
**Purpose**: Checks if a CPO returns a real scalar for given arguments.

---

## Concept Usage Patterns

### 1. Algorithm Constraints
```cpp
template<real_vector_c Vector>
auto my_algorithm(Vector& x) {
    auto temp = clone(x);
    auto& temp_ref = deref_if_needed(temp);
    scale_in_place(temp_ref, 2.0);
    return inner_product(x, temp_ref);
}
```

### 2. Template Specialization
```cpp
template<typename T>
struct vector_traits;

template<real_vector_c V>
struct vector_traits<V> {
    using scalar_type = inner_product_return_t<V>;
    using size_type = dimension_return_t<V>;
    static constexpr bool is_vector = true;
};
```

### 3. SFINAE-style Constraints
```cpp
template<typename V>
std::enable_if_t<real_vector_c<V>, double>
compute_norm(const V& v) {
    return std::sqrt(inner_product(v, v));
}
```

---

## Advanced Topics

### Concept Composition
RVF concepts can be composed with other concepts:
```cpp
template<typename V>
concept cuda_vector_c = real_vector_c<V> &&
    requires(V v) {
        { v.device_ptr() } -> std::convertible_to<void*>;
        { v.is_on_device() } -> std::same_as<bool>;
    };
```

### Conditional Concepts
```cpp
template<typename V>
concept resizable_vector_c = real_vector_c<V> &&
    requires(V v, dimension_return_t<V> n) {
        v.resize(n);
    };
```

### Performance Considerations
- Concept checks are compile-time only (zero runtime cost)
- Use concepts instead of SFINAE for better error messages
- Concept subsumption helps with overload resolution

### Common Pitfalls
1. **Wrapper Types**: Remember that `clone()` may return wrappers; use `deref_if_needed()`
2. **Const Correctness**: Concepts check exact signatures; const mismatches will fail
3. **Template Argument Deduction**: Concepts are checked before implicit conversions

---

## Migration Guide

### From Type Traits to Concepts
```cpp
// Old SFINAE style
template<typename V>
std::enable_if_t<is_vector_v<V>, void>
old_function(V& v) { /* ... */ }

// New concept style
template<real_vector_c V>
void new_function(V& v) { /* ... */ }
```

### Custom Concept Integration
```cpp
// Define domain-specific concepts
template<typename T>
concept finite_element_vector_c = real_vector_c<T> &&
    requires(T v) {
        { v.dof_count() } -> std::same_as<size_t>;
        { v.mesh_info() } -> std::convertible_to<MeshInfo>;
    };

// Use in algorithms
template<finite_element_vector_c Vec>
auto solve_fem_system(const Vec& initial) { /* ... */ }
```
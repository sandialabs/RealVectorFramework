# RealVectorFramework (RVF)

A **header-only, generic C++20 framework** for vector operations and numerical algorithms built on [TInCuP](https://github.com/tincup-org/TInCuP) customization points. RVF provides a unified interface for linear algebra operations that works with any vector-like type, enabling high-performance numerical computing with pluggable backends.

## Key Features

- 🔧 **Generic by Design**: Works with any vector type (`std::vector`, custom types, GPU vectors)
- ⚡ **Zero-Overhead Abstractions**: Compile-time polymorphism via C++20 concepts  
- 🧩 **Pluggable Backends**: Easy integration with specialized libraries (BLAS, CUDA, etc.)
- 📊 **Rich Algorithm Suite**: Trust region methods, conjugate gradient, optimization algorithms
- 🏗️ **Memory Management**: Advanced memory arena system for high-performance computing
- 🎯 **Concept-Driven**: Strong type safety with expressive concepts

---

## Quick Start

```cpp
#include "rvf.hpp"
#include "operations/std_cpo_impl.hpp"  // Enable std::vector support

using namespace rvf;
using Vector = std::vector<double>;

// Create and manipulate vectors using RVF operations
Vector x = {1.0, 2.0, 3.0};
auto y_clone = clone(x);           // Clone vector
auto& y = deref_if_needed(y_clone); // Get reference

scale_in_place(y, 2.0);            // y *= 2.0  
axpy_in_place(y, 1.5, x);          // y += 1.5 * x
double dot = inner_product(x, y);   // Compute x · y
```

- **Documentation**: See `docs/CPO_INTEGRATION_GUIDE.md` for detailed guidance on ADL `tag_invoke` vs `tincup::cpo_impl` specializations

---

## Core Customization Point Objects (CPOs)

RVF provides a complete set of vector operations through **Customization Point Objects** that can be specialized for any vector type:

### 📐 **Basic Operations**
| CPO | Description | Usage |
|-----|-------------|-------|
| `clone(v)` | Create a copy of vector `v` | `auto v2 = clone(v1);` |
| `dimension(v)` | Get vector size/dimension | `size_t n = dimension(v);` |
| `inner_product(x, y)` | Compute dot product x · y | `double dot = inner_product(x, y);` |
| `deref_if_needed(v)` | Dereference wrapper types | `auto& ref = deref_if_needed(clone(v));` |

### ➕ **In-Place Operations**
| CPO | Description | Mathematical Equivalent |
|-----|-------------|-------------------------|
| `scale_in_place(v, α)` | Scale vector by scalar | `v ← α * v` |
| `add_in_place(y, x)` | Vector addition | `y ← y + x` |
| `axpy_in_place(y, α, x)` | Scaled addition | `y ← y + α * x` |

### 🔄 **Advanced Operations**  
| CPO | Description | Purpose |
|-----|-------------|---------|
| `self_map_c<F, Vec>` | Function maps vector to itself | Concept for operators like Hessians |
| `unary_in_place(v, f)` | Apply unary function element-wise | Custom transformations |
| `binary_in_place(z, x, y, f)` | Apply binary function element-wise | Element-wise operations |
| `variadic_in_place(v, f, args...)` | Variadic element-wise operations | Complex transformations |

### 🏗️ **Memory Arena Integration**
| CPO | Description | Use Case |
|-----|-------------|----------|
| `arena_integration` | Memory pool allocation | High-frequency allocations |
| `arena_observers` | Memory usage tracking | Performance analysis |

## Cloning Idiom (Owner + Reference)

Many RVF operations accept and return either values or smart-pointer-like wrappers. The `clone` CPO may therefore return either a value (e.g., `std::vector<double>`) or a wrapper (e.g., `std::unique_ptr<std::vector<double>>`). Use `deref_if_needed` to obtain a reference to the underlying vector.

Recommended idiom: keep the owner and the reference adjacent to avoid lifetime issues and to satisfy concept-constrained APIs (which check types before applying conversions).

```cpp
auto cl = rvf::clone(x); auto& xr = rvf::deref_if_needed(cl);
// use xr as a regular vector; cl owns storage so xr stays valid
```

Rationale:
- Concept checks run before implicit conversions, so passing temporary wrappers to functions constrained on `real_vector_c` will fail. Binding a reference (`xr`) avoids this.
- Keeping the owner (`cl`) and the reference (`xr`) adjacent makes the code read more mathematically while preventing dangling references.

This idiom is used throughout the algorithms in this repo (e.g., Conjugate Gradient, Sherman–Morrison, gradient descent with bounds).

---

## Algorithms & Solvers

RVF provides a comprehensive suite of numerical algorithms for optimization and linear algebra:

### 🎯 **Optimization Algorithms**

#### **Trust Region Methods**
- **`TruncatedCG`**: Steihaug-Toint truncated conjugate gradient for trust region subproblems
  ```cpp
  // Create solver instance (clones objective once)
  TruncatedCG<Objective, Vector> tcg_solver(objective, x_template);
  
  // Solve trust region subproblem: min g^T s + (1/2) s^T H s, ||s|| ≤ δ
  auto result = tcg_solver.solve(x_current, step, trust_radius, params);
  
  // Check termination status
  switch (result.status) {
    case TerminationStatus::CONVERGED: /* ... */
    case TerminationStatus::NEGATIVE_CURVATURE: /* ... */
    case TerminationStatus::TRUST_REGION_BOUNDARY: /* ... */
  }
  ```

#### **Line Search Methods**  
- **`gradient_descent_bounds`**: Projected gradient descent with bound constraints and backtracking line search
  ```cpp
  // Define bound constraints
  bound_constraints<Vector> bounds{lower, upper};
  
  // Solve: min f(x) subject to lower ≤ x ≤ upper  
  gradient_descent_bounds(objective, bounds, x, grad_tol, step_size, max_iter);
  ```

### 🔢 **Linear Algebra Algorithms**

#### **Conjugate Gradient**
- **`conjugate_gradient`**: Iterative solver for symmetric positive definite systems
  ```cpp
  // Solve A*x = b using conjugate gradient
  auto result = conjugate_gradient(A_operator, b, x_initial, tolerance, max_iter);
  ```

#### **Sherman-Morrison Formula**
- **`sherman_morrison_solve`**: Efficient rank-1 update to matrix inverse
  ```cpp
  // Solve (A + u*v^T)*x = b efficiently when A^{-1} is known
  sherman_morrison_solve(A_inv_action, u, v, b, x);
  ```

### 📊 **Algorithm Features**

| Algorithm | Problem Type | Key Features |
|-----------|-------------|--------------|
| `TruncatedCG` | Trust region subproblems | Negative curvature handling, boundary detection |
| `gradient_descent_bounds` | Bound-constrained optimization | Projection, backtracking line search |
| `conjugate_gradient` | Linear systems (SPD) | Iterative, matrix-free capable |
| `sherman_morrison` | Low-rank matrix updates | O(n) complexity for rank-1 updates |

---

## Objective Function Framework

RVF provides a **concept-driven framework** for defining objective functions with automatic differentiation support:

### 🎯 **Core Concepts**
```cpp
// Objective function concepts (output arguments first)
template<typename F, typename Vec>
concept objective_value_c = requires(const F& f, const Vec& x) {
  { f.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
};

template<typename F, typename Vec>  
concept objective_gradient_c = requires(const F& f, const Vec& x, Vec& g) {
  { f.gradient(g, x) } -> std::same_as<void>;  // g = ∇f(x)
};

template<typename F, typename Vec>
concept objective_hess_vec_c = requires(const F& f, const Vec& x, const Vec& v, Vec& hv) {  
  { f.hessVec(hv, v, x) } -> std::same_as<void>;  // hv = H*v where H = ∇²f(x)
};
```

### 📈 **Example: Zakharov Test Function**
```cpp
// f(x) = x^T x + (1/4)(k^T x)^2 + (1/16)(k^T x)^4
auto zakharov = make_zakharov_objective(k_vector);

Vector x = {1.0, 2.0, 3.0};
Vector grad(3), hess_vec_result(3), direction(3);

double f_val = zakharov.value(x);           // Evaluate f(x)  
zakharov.gradient(grad, x);                 // ∇f(x) → grad
zakharov.hessVec(hess_vec_result, direction, x);  // H*direction → result
```

## Examples & Tutorials

RVF includes comprehensive examples demonstrating various features and use cases:

### 🚀 **Getting Started Examples**
- **`examples/main.cpp`**: Basic RVF usage with generic `tag_invoke` implementations
- **`examples/traits_vs_ranges.cpp`**: Backend specialization using `operations/std_cpo_impl.hpp`

### 🧮 **Algorithm Demonstrations**  
- **`examples/conjugate_gradient.cpp`**: Iterative linear system solver
- **`examples/sherman_morrison.cpp`**: Efficient low-rank matrix updates
- **`examples/hs1_gradient_descent.cpp`**: Bound-constrained optimization (Hock-Schittkowski problem)
- **`examples/hs1_advanced.cpp`**: Advanced optimization with iteration callbacks

### 🎯 **Trust Region & Nonlinear Optimization**
- **`examples/zakharov_example.cpp`**: Complete trust region method with truncated CG
- **`examples/zakharov_rvf.cpp`**: RVF-native implementation 
- **`examples/zakharov_truncated_cg.cpp`**: Detailed truncated CG algorithm showcase
- **`examples/zakharov_truncated_cg_simple.cpp`**: Simplified TruncatedCG usage

### 🏗️ **Memory Management**
- **`examples/memory_arena_example.cpp`**: High-performance memory arena usage

### 💡 **Complete Example: Trust Region Optimization**
```cpp
#include "rvf.hpp"
#include "operations/std_cpo_impl.hpp"

using namespace rvf;
using Vector = std::vector<double>;

int main() {
  // Define problem size and initial point
  const size_t n = 5;
  Vector x(n, 3.0);  // Start at x = (3, 3, 3, 3, 3)
  
  // Create Zakharov objective function
  Vector k(n);
  std::iota(k.begin(), k.end(), 1.0);  // k = (1, 2, 3, 4, 5)
  auto objective = make_zakharov_objective(k);
  
  // Set up TruncatedCG solver (efficient: clones only once)
  TruncatedCG<decltype(objective), Vector> tcg_solver(objective, x);
  
  // Trust region parameters
  double trust_radius = 1.0;
  typename decltype(tcg_solver)::Params tcg_params;
  tcg_params.abs_tol = 1e-8;
  tcg_params.rel_tol = 1e-2;
  tcg_params.max_iter = n;
  
  // Solve trust region subproblem
  Vector step(n, 0.0);
  auto result = tcg_solver.solve(x, step, trust_radius, tcg_params);
  
  // Process results
  std::cout << "Status: " << static_cast<int>(result.status) << "\n";
  std::cout << "Iterations: " << result.iter << "\n";
  std::cout << "Step norm: " << result.step_norm << "\n";
  std::cout << "Predicted reduction: " << result.pred_reduction << "\n";
  
  return 0;
}
```

---

## Build & Installation

### **Requirements**
- **C++20 compatible compiler** (GCC 10+, Clang 12+, MSVC 2022+)
- **CMake 3.14+**

### **Build Instructions**
```bash
# Configure build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build all examples and benchmarks  
cmake --build build -j

# Run examples
./build/examples/zakharov_example
./build/examples/conjugate_gradient
./build/examples/memory_arena_example

# Run benchmarks
./build/benchmarks/arena_benchmarks
./build/benchmarks/stack_vs_heap_benchmark
```

### **Integration with Your Project**
RVF is **header-only**, so you can simply copy the `include/` directory or use CMake:

```cmake
# Add RVF as subdirectory
add_subdirectory(RealVectorFramework)
target_link_libraries(your_target PRIVATE rvf::rvf)

# Or use FetchContent
include(FetchContent)
FetchContent_Declare(rvf GIT_REPOSITORY https://github.com/your-repo/RealVectorFramework)
FetchContent_MakeAvailable(rvf)
```

---

## Testing & Benchmarking

### 🧪 **Testing Framework**
RVF includes comprehensive tests for all CPOs and algorithms:

```bash
# Build and run tests
cmake --build build --target test
ctest --test-dir build --verbose

# Run specific test categories
ctest --test-dir build -R "cpo_tests"     # CPO functionality
ctest --test-dir build -R "algorithm"    # Algorithm correctness  
ctest --test-dir build -R "concept"      # Concept validation
```

### ⚡ **Performance Benchmarks**
Benchmark suite for performance analysis and optimization:

#### **Memory Arena Benchmarks**
```bash
./build/benchmarks/arena_benchmarks
```
- Memory pool allocation vs. standard allocation
- Arena lifetime management
- High-frequency allocation patterns

#### **Stack vs Heap Benchmarks**  
```bash
./build/benchmarks/stack_vs_heap_benchmark
```
- Compares different memory allocation strategies
- Vector operation performance analysis
- Cache locality impact measurements

### 📊 **Benchmark Results**
Typical performance improvements with memory arenas:
- **10-100x faster** allocation for small vectors
- **Reduced memory fragmentation** in iterative algorithms  
- **Better cache locality** for numerical computations

---

## Advanced Features

### 🔧 **Custom Vector Types**
Extend RVF to work with your vector types by implementing the required CPOs:

```cpp
// Example: Custom GPU vector type
class GPUVector { /* ... */ };

// Implement required CPOs via tag_invoke
auto tag_invoke(rvf::clone_ftor, const GPUVector& v) -> GPUVector {
  return GPUVector(v);  // GPU memory copy
}

auto tag_invoke(rvf::inner_product_ftor, const GPUVector& x, const GPUVector& y) -> double {
  return gpu_dot_product(x, y);  // CUDA kernel call
}

// Now GPUVector works with all RVF algorithms!
conjugate_gradient(gpu_matrix, gpu_b, gpu_x, tolerance);
```

### 🏗️ **Memory Arena Integration**
For high-performance computing scenarios:

```cpp
#include "operations/memory_arena.hpp"

// Create memory arena for temporary allocations
rvf::MemoryArena arena(1024 * 1024);  // 1MB pool

// Use arena in iterative algorithms (automatic cleanup)
{
  auto scoped_arena = arena.create_scope();
  
  // All vector clones use arena memory
  auto temp1 = clone(large_vector);  // Arena allocation
  auto temp2 = clone(another_vector); // Arena allocation
  
  // Efficient iterative algorithm
  for (int i = 0; i < 1000; ++i) {
    conjugate_gradient_step(temp1, temp2, /* ... */);
    // No malloc/free overhead!
  }
  
} // Arena memory automatically reclaimed
```

### 🎯 **Concept Customization**
Define problem-specific concepts for type safety:

```cpp
// Domain-specific concept
template<typename T>
concept finite_element_vector_c = real_vector_c<T> && requires(T v) {
  { v.dof_count() } -> std::same_as<size_t>;
  { v.mesh_info() } -> std::convertible_to<MeshInfo>;
};

// Algorithm that works only with FE vectors
template<finite_element_vector_c Vec>
auto finite_element_solver(const Vec& initial_guess) {
  // Implementation using RVF operations
  auto residual = clone(initial_guess);
  scale_in_place(residual, 0.0);
  // ...
}
```

---

## Contributing & License

### 🤝 **Contributing**
Contributions are welcome! Please see our contribution guidelines for:
- Code style and formatting
- Test requirements  
- Documentation standards
- Performance benchmarking

### 📄 **License**
This project is developed by **Sandia National Laboratories** under the terms of Contract DE-NA0003525 with NTESS. The U.S. Government retains certain rights in this software.

**Questions?** Contact Greg von Winckel (gvonwin@sandia.gov)

---

## Related Projects

- **[TInCuP](https://github.com/tincup-org/TInCuP)**: The customization point framework powering RVF
- **[tabulate](https://github.com/p-ranav/tabulate)**: Table formatting library used in examples

---

*RealVectorFramework - Bridging generic programming and high-performance numerical computing* 🚀

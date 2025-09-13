# RVF Algorithms Reference

This document provides comprehensive reference for all algorithms implemented in RVF, organized by mathematical domain and application area.

## Overview

RVF algorithms are built on the foundation of CPOs and concepts, providing generic implementations that work with any `real_vector_c` compliant type. All algorithms support:

- Generic vector types through concepts
- Customizable precision and convergence criteria
- Comprehensive result reporting
- Memory arena integration for performance

---

## Linear Algebra Algorithms

### Conjugate Gradient Method

#### `conjugate_gradient`

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

**Purpose**: Solves linear system `A*x = b` using conjugate gradient iteration for symmetric positive definite matrices.

**Template Parameters**:
- `LinearOperator`: Function-like object representing matrix A
- `Vector`: Type satisfying `real_vector_c`

**Parameters**:
- `A`: Linear operator (must support `A(result, input)`)
- `b`: Right-hand side vector
- `x`: Initial guess (modified to solution)
- `tolerance`: Convergence tolerance (default: 1e-8)
- `max_iterations`: Maximum iterations (-1 for automatic: `dimension(b)`)

**Returns**: `CGResult` structure containing convergence information

**Mathematical Background**:
- Solves `A*x = b` where A is symmetric positive definite
- Minimizes quadratic function `½x^T A x - b^T x`
- Theoretically converges in at most n steps (n = dimension)
- Practical convergence often much faster

**Linear Operator Interface**:
```cpp
struct MatrixOperator {
    template<typename VecIn, typename VecOut>
    void operator()(VecOut& result, const VecIn& input) const {
        // result = A * input
    }
};
```

**CGResult Structure**:
```cpp
struct CGResult {
    bool converged;           // True if converged within tolerance
    int iterations;           // Number of iterations performed
    double final_residual;    // Final residual norm
    double initial_residual;  // Initial residual norm
};
```

**Algorithm Details**:
1. Initialize residual `r₀ = b - A*x₀`
2. Set initial search direction `p₀ = r₀`
3. For k = 0, 1, 2, ... until convergence:
   - `αₖ = (rₖ^T rₖ) / (pₖ^T A pₖ)`
   - `xₖ₊₁ = xₖ + αₖ pₖ`
   - `rₖ₊₁ = rₖ - αₖ A pₖ`
   - `βₖ = (rₖ₊₁^T rₖ₊₁) / (rₖ^T rₖ)`
   - `pₖ₊₁ = rₖ₊₁ + βₖ pₖ`

**Example Usage**:
```cpp
// Example: Solve tridiagonal system
struct TridiagonalOperator {
    double a, b, c;  // subdiagonal, diagonal, superdiagonal

    template<typename VecIn, typename VecOut>
    void operator()(VecOut& result, const VecIn& x) const {
        size_t n = dimension(x);
        fill(result, 0.0);

        for (size_t i = 0; i < n; ++i) {
            result[i] += b * x[i];  // diagonal
            if (i > 0) result[i] += a * x[i-1];      // subdiagonal
            if (i < n-1) result[i] += c * x[i+1];    // superdiagonal
        }
    }
};

// Solve system
std::vector<double> b(100, 1.0);  // RHS
std::vector<double> x(100, 0.0);  // Initial guess

TridiagonalOperator A{-1.0, 2.0, -1.0};
auto result = conjugate_gradient(A, b, x, 1e-10);

std::cout << "Converged: " << result.converged << "\n";
std::cout << "Iterations: " << result.iterations << "\n";
std::cout << "Final residual: " << result.final_residual << "\n";
```

**Convergence Theory**:
- Error bound: `‖eₖ‖_A ≤ 2 * (√κ - 1)/(√κ + 1))^k * ‖e₀‖_A`
- κ = condition number of A
- Well-conditioned problems converge rapidly

**Performance Considerations**:
- Memory: O(n) additional storage for residual and search direction
- Operations per iteration: 1 matrix-vector product, 3 vector updates, 2 inner products
- Cache-friendly: mostly vector operations
- Parallelizable: matrix-vector product and vector operations

---

### Sherman-Morrison Formula

#### `sherman_morrison_solve`

```cpp
template<typename LinearOperator, typename Vector>
void sherman_morrison_solve(
    const LinearOperator& A_inv_action,
    const Vector& u, const Vector& v,
    const Vector& b, Vector& x
);
```

**Purpose**: Efficiently solves `(A + u*v^T)*x = b` when action of `A^(-1)` is available.

**Mathematical Formula**:
`(A + u*v^T)^(-1) = A^(-1) - (A^(-1)*u*v^T*A^(-1)) / (1 + v^T*A^(-1)*u)`

**Template Parameters**:
- `LinearOperator`: Represents action of `A^(-1)`
- `Vector`: Type satisfying `real_vector_c`

**Parameters**:
- `A_inv_action`: Function-like object for `A^(-1) * vector`
- `u`, `v`: Rank-1 update vectors
- `b`: Right-hand side vector
- `x`: Solution vector (output)

**Algorithm Steps**:
1. Compute `z₁ = A^(-1) * u`
2. Compute `z₂ = A^(-1) * b`
3. Compute `σ = v^T * z₁`
4. Check `1 + σ ≠ 0` (invertibility condition)
5. Compute `τ = v^T * z₂`
6. Set `x = z₂ - (τ/(1 + σ)) * z₁`

**Complexity**: O(n) if `A^(-1)` action is O(n), plus 2 applications of `A^(-1)`

**Example Usage**:
```cpp
// Example: Diagonal matrix with rank-1 update
struct DiagonalInverse {
    std::vector<double> diag_inv;

    template<typename VecIn, typename VecOut>
    void operator()(VecOut& result, const VecIn& input) const {
        for (size_t i = 0; i < dimension(input); ++i) {
            result[i] = diag_inv[i] * input[i];
        }
    }
};

std::vector<double> diagonal{2, 3, 4, 5};
std::vector<double> diag_inv(diagonal.size());
std::transform(diagonal.begin(), diagonal.end(), diag_inv.begin(),
               [](double x) { return 1.0/x; });

DiagonalInverse A_inv{diag_inv};
std::vector<double> u{1, 1, 1, 1};  // Rank-1 update vector
std::vector<double> v{1, 0, 0, 1};  // Rank-1 update vector
std::vector<double> b{10, 15, 20, 25}; // RHS
std::vector<double> x(4);

sherman_morrison_solve(A_inv, u, v, b, x);
// x now contains solution to (diag(diagonal) + u*v^T) * x = b
```

**Numerical Stability**:
- Check `|1 + v^T * A^(-1) * u| > ε` for some small ε
- If denominator is near zero, matrix is nearly singular
- Consider iterative refinement for ill-conditioned problems

**Applications**:
- Updating matrix factorizations
- Sequential data processing (adding one data point)
- Optimization algorithms with rank-1 Hessian updates
- Kalman filtering

---

## Optimization Algorithms

### Trust Region Methods

#### `TruncatedCG` Class

```cpp
template<typename Objective, typename Vector>
class TruncatedCG {
public:
    struct Params {
        double abs_tol = 1e-8;        // Absolute tolerance
        double rel_tol = 1e-2;        // Relative tolerance
        int max_iter = -1;            // Max iterations (-1 = auto)
        bool verbose = false;         // Enable logging
    };

    struct Result {
        TerminationStatus status;     // Termination reason
        int iter;                     // Iterations performed
        double step_norm;             // Norm of computed step
        double pred_reduction;        // Predicted reduction
        double cauchy_reduction;      // Cauchy point reduction
    };

    // Constructor
    TruncatedCG(const Objective& obj, const Vector& template_vec);

    // Solve trust region subproblem
    auto solve(const Vector& x_current, Vector& step,
               double trust_radius, const Params& params = {}) -> Result;
};
```

**Purpose**: Solves trust region subproblem using Steihaug-Toint truncated conjugate gradient.

**Trust Region Subproblem**:
Minimize `g^T*s + ½*s^T*H*s` subject to `‖s‖ ≤ δ`
where:
- `g` = gradient at current point
- `H` = Hessian at current point
- `δ` = trust radius
- `s` = step to compute

**Termination Conditions**:
```cpp
enum class TerminationStatus {
    CONVERGED,                 // Gradient tolerance satisfied
    NEGATIVE_CURVATURE,        // Encountered negative curvature
    TRUST_REGION_BOUNDARY,     // Hit trust region boundary
    MAX_ITERATIONS,           // Iteration limit reached
    NUMERICAL_ERROR           // Numerical issues detected
};
```

**Constructor Parameters**:
- `obj`: Objective function (must satisfy appropriate concepts)
- `template_vec`: Template vector for memory allocation

**Solve Parameters**:
- `x_current`: Current optimization point
- `step`: Output step vector (modified in-place)
- `trust_radius`: Trust region radius δ
- `params`: Algorithm parameters

**Objective Function Requirements**:
```cpp
template<typename F, typename Vec>
concept trust_region_objective_c =
    objective_gradient_c<F, Vec> &&      // f.gradient(g, x)
    objective_hess_vec_c<F, Vec>;        // f.hessVec(hv, v, x)
```

**Algorithm Overview** (Steihaug-Toint):
1. Initialize `s₀ = 0`, `r₀ = g`, `p₀ = -g`
2. For k = 0, 1, 2, ... until convergence:
   - Compute `Hₖpₖ` (Hessian-vector product)
   - Check curvature: if `pₖ^T Hₖ pₖ ≤ 0`, find boundary solution
   - Compute step length `αₖ`
   - Check trust region: if `‖sₖ + αₖpₖ‖ ≥ δ`, find boundary intersection
   - Update: `sₖ₊₁ = sₖ + αₖpₖ`, `rₖ₊₁ = rₖ + αₖHₖpₖ`
   - Check convergence: if `‖rₖ₊₁‖ ≤ tol`, terminate
   - Update search direction: `pₖ₊₁ = -rₖ₊₁ + βₖpₖ`

**Example Usage**:
```cpp
// Define objective function
auto objective = make_zakharov_objective(k_vector);

// Create solver (efficient: clones objective only once)
Vector x_template(n);
TruncatedCG<decltype(objective), Vector> solver(objective, x_template);

// Set parameters
typename TruncatedCG<decltype(objective), Vector>::Params params;
params.abs_tol = 1e-8;
params.rel_tol = 1e-2;
params.max_iter = n;

// Solve subproblem
Vector step(n, 0.0);
auto result = solver.solve(x_current, step, trust_radius, params);

// Process results
switch (result.status) {
    case TerminationStatus::CONVERGED:
        std::cout << "Converged in " << result.iter << " iterations\n";
        break;
    case TerminationStatus::NEGATIVE_CURVATURE:
        std::cout << "Negative curvature detected\n";
        break;
    case TerminationStatus::TRUST_REGION_BOUNDARY:
        std::cout << "Hit trust region boundary\n";
        break;
}
```

**Performance Features**:
- Efficient memory management (clones objective once)
- Minimal vector allocations during solve
- Cache-friendly vector operations
- Support for memory arenas

---

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

**Purpose**: Solves bound-constrained optimization using projected gradient descent with backtracking line search.

**Problem Formulation**:
Minimize `f(x)` subject to `lower ≤ x ≤ upper` (element-wise)

**Template Parameters**:
- `Objective`: Function object satisfying `objective_value_c` and `objective_gradient_c`
- `Vector`: Type satisfying `real_vector_c`

**Parameters**:
- `objective`: Objective function to minimize
- `bounds`: Box constraints specification
- `x`: Starting point (modified to solution)
- `grad_tolerance`: Convergence tolerance on projected gradient norm
- `initial_step_size`: Initial step size for line search
- `max_iterations`: Maximum optimization iterations

**Bound Constraints Structure**:
```cpp
template<typename Vector>
struct bound_constraints {
    Vector lower;    // Lower bounds
    Vector upper;    // Upper bounds

    // Constructor for uniform bounds
    bound_constraints(double lb, double ub, size_t n);

    // Constructor for element-wise bounds
    bound_constraints(const Vector& lb, const Vector& ub);

    // Projection onto feasible region
    void project(Vector& x) const;
};
```

**Algorithm Steps**:
1. Evaluate gradient at current point
2. Compute projected gradient direction
3. Check convergence (projected gradient norm)
4. Perform backtracking line search:
   - Try full step with projection
   - If Armijo condition not satisfied, reduce step size
   - Repeat until acceptable step found
5. Update current point with projection
6. Repeat until convergence

**Armijo Line Search Condition**:
`f(P(x - α∇f(x))) ≤ f(x) + c₁α∇f(x)^T(P(x - α∇f(x)) - x)`
where P(·) is projection onto feasible region, c₁ ∈ (0, 1).

**Example Usage**:
```cpp
// Define bounded optimization problem
auto objective = make_simple_quadratic();
size_t n = 10;

// Set box constraints: -5 ≤ x ≤ 5
bound_constraints<std::vector<double>> bounds(-5.0, 5.0, n);

// Starting point
std::vector<double> x(n, 3.0);  // Start at x = (3, 3, ..., 3)

// Solve
gradient_descent_bounds(
    objective, bounds, x,
    1e-6,    // gradient tolerance
    1.0,     // initial step size
    1000     // max iterations
);

std::cout << "Optimal point: ";
for (double xi : x) std::cout << xi << " ";
std::cout << "\nObjective value: " << objective.value(x) << "\n";
```

**Convergence Theory**:
- Converges to stationary point (KKT conditions satisfied)
- Rate depends on objective function properties
- Linear convergence for strongly convex objectives

**Practical Considerations**:
- Works well for smooth objectives
- Handles simple bound constraints efficiently
- May be slow for ill-conditioned problems
- Consider preconditioning for better performance

---

## Objective Function Framework

### Core Concepts

#### `objective_value_c`
```cpp
template<typename F, typename Vec>
concept objective_value_c = requires(const F& f, const Vec& x) {
    { f.value(x) } -> std::convertible_to<vector_value_t<Vec>>;
};
```

#### `objective_gradient_c`
```cpp
template<typename F, typename Vec>
concept objective_gradient_c = requires(const F& f, const Vec& x, Vec& g) {
    { f.gradient(g, x) } -> std::same_as<void>;
};
```

#### `objective_hess_vec_c`
```cpp
template<typename F, typename Vec>
concept objective_hess_vec_c = requires(const F& f, const Vec& x, const Vec& v, Vec& hv) {
    { f.hessVec(hv, v, x) } -> std::same_as<void>;
};
```

### Test Problems

#### Zakharov Function

```cpp
template<typename Vector>
auto make_zakharov_objective(const Vector& k) -> /* implementation-defined */;
```

**Mathematical Definition**:
`f(x) = x^T*x + (1/4)*(k^T*x)^2 + (1/16)*(k^T*x)^4`

**Properties**:
- Global minimum at x = 0 with f(0) = 0
- Non-convex due to quartic term
- Gradient: `∇f(x) = 2x + (1/2)(k^T*x)*k + (1/4)(k^T*x)^3*k`
- Hessian: `H = 2I + (1/2)*k*k^T + (3/4)*(k^T*x)^2*k*k^T`

**Usage**:
```cpp
size_t n = 5;
Vector k(n);
std::iota(k.begin(), k.end(), 1.0);  // k = [1, 2, 3, 4, 5]

auto zakharov = make_zakharov_objective(k);

Vector x(n, 1.0);
Vector grad(n), hess_vec(n), direction(n);

double f_val = zakharov.value(x);
zakharov.gradient(grad, x);
zakharov.hessVec(hess_vec, direction, x);
```

---

## Algorithm Composition

### Combining Algorithms

RVF algorithms can be combined to create sophisticated optimization methods:

```cpp
template<typename Objective, typename Vector>
class TrustRegionMethod {
    TruncatedCG<Objective, Vector> cg_solver;
    double trust_radius = 1.0;

public:
    auto solve(const Objective& obj, Vector& x) -> OptimizationResult {
        for (int iter = 0; iter < max_outer_iter; ++iter) {
            // Solve trust region subproblem
            Vector step(dimension(x), 0.0);
            auto cg_result = cg_solver.solve(x, step, trust_radius);

            // Evaluate actual vs predicted reduction
            double actual_red = obj.value(x) - obj.value_at_step(x, step);
            double pred_red = cg_result.pred_reduction;

            double ratio = actual_red / pred_red;

            // Update trust radius and accept/reject step
            if (ratio > 0.75) {
                trust_radius = std::min(2 * trust_radius, max_trust_radius);
                axpy_in_place(x, 1.0, step);  // Accept step
            } else if (ratio > 0.25) {
                axpy_in_place(x, 1.0, step);  // Accept step, keep radius
            } else {
                trust_radius *= 0.5;  // Reject step, reduce radius
            }

            // Check convergence
            if (l2norm(gradient) < tolerance) break;
        }
    }
};
```

### Memory Arena Integration

All algorithms support memory arena integration for high-performance computing:

```cpp
#include "operations/memory/memory_arena.hpp"

void high_performance_solve() {
    rvf::MemoryArena arena(1024 * 1024);  // 1MB pool

    {
        auto scoped_arena = arena.create_scope();

        // All vector operations use arena memory
        auto result = conjugate_gradient(A, b, x);

        // Temporary vectors automatically managed
        for (int iter = 0; iter < 1000; ++iter) {
            auto temp = clone(x);  // Arena allocation
            // ... algorithm steps
        }

    } // Arena memory automatically reclaimed
}
```
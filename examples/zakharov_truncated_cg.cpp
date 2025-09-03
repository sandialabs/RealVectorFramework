/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

Example: Using Truncated CG to minimize the Zakharov objective function
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <tabulate/table.hpp>

// We'll implement a simple std::vector-based implementation that satisfies real_vector_c
// This demonstrates how to use RVF concepts with concrete types

// Helper function for formatting doubles
std::string format_double(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

// Include RVF first to get the proper declarations
#define TINCUP_STATIC_ASSERTS_DISABLE
#include <rvf.hpp>

namespace rvf {

// Deref implementation for std::vector (no-op)
template<typename T>
constexpr auto tag_invoke(deref_if_needed_ftor, std::vector<T>& v) -> std::vector<T>& {
  return v;
}

template<typename T>
constexpr auto tag_invoke(deref_if_needed_ftor, const std::vector<T>& v) -> const std::vector<T>& {
  return v;
}

// Add in place: y += x
template<typename T>
constexpr auto tag_invoke(add_in_place_ftor, std::vector<T>& y, const std::vector<T>& x) -> void {
  for (size_t i = 0; i < y.size(); ++i) {
  y[i] += x[i];
  }
}

// Clone: create a copy
template<typename T>
constexpr auto tag_invoke(clone_ftor, const std::vector<T>& x) -> std::vector<T> {
  return x; // Copy constructor
}

// Dimension: return size
template<typename T>
constexpr auto tag_invoke(dimension_ftor, const std::vector<T>& x) -> size_t {
  return x.size();
}

// Inner product: dot product
template<typename T>
constexpr auto tag_invoke(inner_product_ftor, const std::vector<T>& x, const std::vector<T>& y) -> T {
  T result = T(0);
  for (size_t i = 0; i < x.size(); ++i) {
  result += x[i] * y[i];
  }
  return result;
}

// Scale in place: x *= alpha
template<typename T, typename S>
constexpr auto tag_invoke(scale_in_place_ftor, std::vector<T>& x, S alpha) -> void {
  for (auto& xi : x) {
  xi *= alpha;
  }
}

// AXPY in place: y += alpha * x
template<typename T, typename S>
constexpr auto tag_invoke(axpy_in_place_ftor, std::vector<T>& y, S alpha, const std::vector<T>& x) -> void {
  for (size_t i = 0; i < y.size(); ++i) {
  y[i] += alpha * x[i];
  }
}

} // namespace rvf

using Vector = std::vector<double>;

// Specialization of create_index_vector_impl for std::vector
namespace rvf {
template<>
void create_index_vector_impl<Vector>(Vector& k, vector_size_t<Vector> n) {
  k.resize(n);
  for (size_t i = 0; i < n; ++i) {
  k[i] = static_cast<double>(i + 1);
  }
}
}

// Simple trust region method using Truncated CG
class trust_region_solver {
private:
  double delta_; // Trust region radius
  double eta1_, eta2_; // Trust region acceptance parameters
  double gamma1_, gamma2_; // Trust region update parameters
  
public:
  trust_region_solver(double delta_init = 1.0, 
		      double eta1 = 0.1, 
		      double eta2 = 0.75,
                      double gamma1 = 0.5, 
		      double gamma2 = 2.0)
  : delta_(delta_init), eta1_(eta1), eta2_(eta2), gamma1_(gamma1), gamma2_(gamma2) {}
  
  template<typename Objective>
  void solve(Objective& obj, Vector& x, int max_iter = 100, double gtol = 1e-6) {
  using namespace rvf;
  
  Vector grad(x.size());
  Vector step(x.size());
  Vector x_new(x.size());
  
  // Store iteration data for final summary
  std::vector<double> reduction_ratios;
  
  // Create iteration results table
  tabulate::Table iteration_table;
  iteration_table.add_row({"Iter", "f(x)", "||grad||", "delta", "TCG_flag", "TCG_iter", "rho"});
  iteration_table.format().font_style({tabulate::FontStyle::bold});
  
  std::cout << "Trust Region Method with Truncated CG for Zakharov Function\n";
  std::cout << "==========================================================\n\n";
  
  for (int iter = 0; iter < max_iter; ++iter) {
    // Evaluate objective and gradient
    double f_x = obj.value(x);
    obj.gradient(grad, x);
    double grad_norm = std::sqrt(inner_product(grad, grad));
    
    // Check convergence
    if (grad_norm < gtol) {
    std::cout << "Converged with gradient norm: " << grad_norm << "\n";
    break;
    }
    
    // Create Hessian operator
    // Create TruncatedCG solver instance
    rvf::TruncatedCG<decltype(obj), Vector> tcg_solver(obj, x);
    
    // Set up Truncated CG parameters
    typename decltype(tcg_solver)::Params tcg_params;
    tcg_params.max_iter = x.size(); // At most n iterations
    tcg_params.abs_tol = 1e-8;
    tcg_params.rel_tol = 0.1;
    
    // Solve trust region subproblem using Truncated CG
    std::fill(step.begin(), step.end(), 0.0); // Initialize step to zero
    auto tcg_result = tcg_solver.solve(x, step, delta_, tcg_params);
    
    // Compute predicted reduction
    double pred_reduction = tcg_result.pred_reduction;
    
    // Evaluate at trial point
    x_new = x;
    axpy_in_place(x_new, 1.0, step); // x_new = x + step
    double f_new = obj.value(x_new);
    
    // Compute actual reduction
    double actual_reduction = f_x - f_new;
    
    // Compute reduction ratio
    double rho = (pred_reduction > 0) ? actual_reduction / pred_reduction : -1.0;
    
    // Add iteration info to table
    iteration_table.add_row({
    std::to_string(iter),
    format_double(f_x, 6),
    format_double(grad_norm, 6), 
    format_double(delta_, 6),
    std::to_string(static_cast<int>(tcg_result.status)),
    std::to_string(tcg_result.iter),
    format_double(rho, 6)
    });
    
    // Store reduction ratio for summary
    reduction_ratios.push_back(rho);
    
    // Trust region update
    if (rho >= eta1_) {
    // Accept step
    x = x_new;
    
    if (rho >= eta2_) {
      // Expand trust region
      delta_ = std::min(gamma2_ * delta_, 10.0); // Cap at reasonable size
    }
    } else {
    // Reject step, contract trust region
    delta_ = gamma1_ * delta_;
    }
    
    // Prevent trust region from becoming too small
    delta_ = std::max(delta_, 1e-12);
  }
  
  // Print iteration table
  std::cout << iteration_table << "\n\n";
  
  // Create final solution table
  tabulate::Table solution_table;
  solution_table.add_row({"Variable", "Value"});
  solution_table.format().font_style({tabulate::FontStyle::bold});
  
  for (size_t i = 0; i < x.size(); ++i) {
    solution_table.add_row({"x[" + std::to_string(i) + "]", format_double(x[i], 8)});
  }
  solution_table.add_row({"f(x) final", format_double(obj.value(x), 8)});
  
  std::cout << "Final Solution:\n" << solution_table << "\n";
  }
};

int main() {
  using namespace rvf;
  
  // Problem dimension
  const size_t n = 5;
  
  // Initial point
  Vector x(n, 3.0); // Start at x = (3, 3, 3, 3, 3)
  
  // Create Zakharov objective
  // First create k vector manually since we need it for construction
  Vector k(n);
  for (size_t i = 0; i < n; ++i) {
  k[i] = static_cast<double>(i + 1);
  }
  
  auto zakharov = make_zakharov_objective(k);
  
  // Create initial conditions table
  tabulate::Table initial_table;
  initial_table.add_row({"Parameter", "Value"});
  initial_table.format().font_style({tabulate::FontStyle::bold});
  initial_table.add_row({"Problem dimension", std::to_string(n)});
  
  std::string initial_point = "(";
  for (size_t i = 0; i < n; ++i) {
  if (i > 0) initial_point += ", ";
  initial_point += std::to_string(x[i]);
  }
  initial_point += ")";
  initial_table.add_row({"Initial point", initial_point});
  initial_table.add_row({"Initial f(x)", format_double(zakharov.value(x), 6)});
  
  std::cout << "Problem Setup:\n" << initial_table << "\n\n";
  
  // Solve using trust region method with Truncated CG
  trust_region_solver solver;
  solver.solve(zakharov, x, 50, 1e-6);
  
  // Create TCG flags reference table
  tabulate::Table flags_table;
  flags_table.add_row({"Flag", "Meaning"});
  flags_table.format().font_style({tabulate::FontStyle::bold});
  flags_table.add_row({"0", "Converged"});
  flags_table.add_row({"1", "Maximum iterations"});
  flags_table.add_row({"2", "Negative curvature encountered"});
  flags_table.add_row({"3", "Trust region boundary reached"});
  flags_table.add_row({"4", "Preconditioner failure"});
  
  std::cout << "Truncated CG Termination Flags:\n" << flags_table << "\n";
  
  // Verify solution (should be close to zero)
  Vector grad(n);
  zakharov.gradient(grad, x);
  double grad_norm = std::sqrt(inner_product(grad, grad));
  // Create verification results table
  tabulate::Table verification_table;
  verification_table.add_row({"Test", "Result"});
  verification_table.format().font_style({tabulate::FontStyle::bold});
  verification_table.add_row({"Final gradient norm", format_double(grad_norm, 8)});
  
  // Test inverse Hessian functionality
  Vector v(n, 1.0); // Test vector
  Vector hv(n), inv_hv(n), should_be_v(n);
  
  zakharov.hessVec(hv, v, x);
  zakharov.inv_hess_vec(x, hv, inv_hv);
  
  // Compute error
  should_be_v = inv_hv;
  axpy_in_place(should_be_v, -1.0, v); // should_be_v = inv_hv - v
  double error = std::sqrt(inner_product(should_be_v, should_be_v));
  verification_table.add_row({"||H^{-1}(H*v) - v|| error", format_double(error, 8)});
  
  std::cout << "Verification Results:\n" << verification_table << "\n";
  
  // Create vector comparison table
  tabulate::Table vector_table;
  vector_table.add_row({"Component", "v (original)", "H^{-1}(H*v)", "Difference"});
  vector_table.format().font_style({tabulate::FontStyle::bold});
  
  for (size_t i = 0; i < n; ++i) {
  double diff = inv_hv[i] - v[i];
  vector_table.add_row({
    std::to_string(i),
    format_double(v[i], 6),
    format_double(inv_hv[i], 6),
    format_double(diff, 8)
  });
  }
  
  std::cout << "Inverse Hessian Test (H^{-1}(H*v) should equal v):\n" << vector_table << "\n";
  
  return 0;
}

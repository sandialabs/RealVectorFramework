/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

Example: Using Truncated CG to minimize the Zakharov objective function with RVF CPOs
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <tabulate/table.hpp>

// Helper function for formatting doubles
std::string format_double(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

// First, we need to provide tag_invoke specializations for std::vector before including RVF
namespace rvf {

// Forward declarations of the CPO types we need to specialize
struct add_in_place_ftor;
struct clone_ftor;
struct dimension_ftor;
struct inner_product_ftor;
struct scale_in_place_ftor;
struct axpy_in_place_ftor;


// Add in place: y += x
template<typename T>
constexpr void tag_invoke(add_in_place_ftor, std::vector<T>& y, const std::vector<T>& x) {
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] += x[i];
  }
}

// Clone: create a copy
template<typename T>
constexpr std::vector<T> tag_invoke(clone_ftor, const std::vector<T>& x) {
  return x; // Copy constructor
}

// Dimension: return size
template<typename T>
constexpr size_t tag_invoke(dimension_ftor, const std::vector<T>& x) noexcept {
  return x.size();
}

// Inner product: dot product
template<typename T>
constexpr T tag_invoke(inner_product_ftor, const std::vector<T>& x, const std::vector<T>& y) {
  return std::inner_product(x.begin(), x.end(), y.begin(), T(0));
}

// Scale in place: x *= alpha
template<typename T, typename S>
constexpr void tag_invoke(scale_in_place_ftor, std::vector<T>& x, S alpha) {
  for (auto& xi : x) {
    xi *= alpha;
  }
}

// AXPY in place: y += alpha * x
template<typename T, typename S>
constexpr void tag_invoke(axpy_in_place_ftor, std::vector<T>& y, S alpha, const std::vector<T>& x) {
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

} // namespace rvf

// Now include the RVF framework
#define TINCUP_STATIC_ASSERTS_DISABLE
#include <core/rvf.hpp>

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

// Trust region solver using RVF components
class trust_region_solver {
private:
  double delta_;
  double eta1_, eta2_;
  double gamma1_, gamma2_;
  
public:
  trust_region_solver(double delta_init = 1.0, double eta1 = 0.1, double eta2 = 0.75,
             double gamma1 = 0.5, double gamma2 = 2.0)
    : delta_(delta_init), eta1_(eta1), eta2_(eta2), gamma1_(gamma1), gamma2_(gamma2) {}
  
  template<typename Objective>
  void solve(Objective& obj, Vector& x, int max_iter = 100, double gtol = 1e-6) {
    using namespace rvf;
    
    auto grad_cl = clone(x); auto& grad = deref_if_needed(grad_cl);
    auto step_cl = clone(x); auto& step = deref_if_needed(step_cl);  
    auto x_new_cl = clone(x); auto& x_new = deref_if_needed(x_new_cl);
    
    // Create iteration results table
    tabulate::Table iteration_table;
    iteration_table.add_row({"Iter", "f(x)", "||grad||", "delta", "TCG_flag", "TCG_iter", "rho"});
    iteration_table.format().font_style({tabulate::FontStyle::bold});
    
    std::cout << "Trust Region Method with Truncated CG for Zakharov Function\n";
    std::cout << "==========================================================\n\n";
    
    for (int iter = 0; iter < max_iter; ++iter) {
      // Evaluate objective and gradient using RVF
      double f_x = obj.value(x);
      obj.gradient(grad, x);
      double grad_norm = std::sqrt(inner_product(grad, grad));
      
      // Check convergence
      if (grad_norm < gtol) {
        std::cout << "Converged with gradient norm: " << grad_norm << "\n";
        break;
      }
      
      // Create TruncatedCG solver instance
      rvf::TruncatedCG tcg_solver(obj, x);
      
      // Set up Truncated CG parameters
      typename decltype(tcg_solver)::Params tcg_params;
      tcg_params.max_iter = x.size();
      tcg_params.abs_tol = 1e-8;
      tcg_params.rel_tol = 0.1;
      
      // Solve trust region subproblem using RVF Truncated CG
      scale_in_place(step, 0.0); // Initialize step to zero
      auto tcg_result = tcg_solver.solve(x, step, delta_, tcg_params);
      
      // Evaluate at trial point using RVF operations
      x_new = x;
      axpy_in_place(x_new, 1.0, step); // x_new = x + step
      double f_new = obj.value(x_new);
      
      // Compute reductions and ratio
      double actual_reduction = f_x - f_new;
      double pred_reduction = tcg_result.pred_reduction;
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
      
      // Trust region update
      if (rho >= eta1_) {
        x = x_new; // Accept step
        if (rho >= eta2_) {
          delta_ = std::min(gamma2_ * delta_, 10.0); // Expand
        }
      } else {
        delta_ = gamma1_ * delta_; // Contract
      }
      
      delta_ = std::max(delta_, 1e-12); // Prevent too small
    }
    
    // Print results using tabulate
    std::cout << iteration_table << "\n\n";
    
    // Final solution table
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
  
  const size_t n = 5;
  Vector x(n, 3.0); // Initial point
  
  // Create k vector manually for Zakharov objective
  Vector k(n);
  for (size_t i = 0; i < n; ++i) {
    k[i] = static_cast<double>(i + 1);
  }
  
  // Create Zakharov objective using RVF
  auto zakharov = make_zakharov_objective(k);
  
  // Problem setup table
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
  
  // Solve using trust region method with RVF components
  trust_region_solver solver;
  solver.solve(zakharov, x, 50, 1e-6);
  
  // TCG flags reference table
  tabulate::Table flags_table;
  flags_table.add_row({"Flag", "Meaning"});
  flags_table.format().font_style({tabulate::FontStyle::bold});
  flags_table.add_row({"0", "Converged"});
  flags_table.add_row({"1", "Maximum iterations"});
  flags_table.add_row({"2", "Negative curvature encountered"});
  flags_table.add_row({"3", "Trust region boundary reached"});
  flags_table.add_row({"4", "Preconditioner failure"});
  
  std::cout << "Truncated CG Termination Flags:\n" << flags_table << "\n";
  
  // Verification using RVF operations
  auto grad_cl = clone(x); auto& grad = deref_if_needed(grad_cl);
  zakharov.gradient(grad, x);
  double grad_norm = std::sqrt(inner_product(grad, grad));
  
  // Test inverse Hessian functionality
  auto v_cl = clone(x); auto& v = deref_if_needed(v_cl);
  auto hv_cl = clone(x); auto& hv = deref_if_needed(hv_cl);
  auto inv_hv_cl = clone(x); auto& inv_hv = deref_if_needed(inv_hv_cl);
  auto error_cl = clone(x); auto& error_vec = deref_if_needed(error_cl);
  
  scale_in_place(v, 0.0);
  axpy_in_place(v, 1.0, x); // v = x (as test vector)
  scale_in_place(v, 0.0);   // Actually, let's use v = (1,1,1,1,1)
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = 1.0;
  }
  
  zakharov.hessVec(hv, v, x);
  zakharov.inv_hess_vec(x, hv, inv_hv);
  
  // Compute error: ||H^{-1}(H*v) - v||
  error_vec = inv_hv;
  axpy_in_place(error_vec, -1.0, v); // error_vec = inv_hv - v
  double error = std::sqrt(inner_product(error_vec, error_vec));
  
  // Verification results table
  tabulate::Table verification_table;
  verification_table.add_row({"Test", "Result"});
  verification_table.format().font_style({tabulate::FontStyle::bold});
  verification_table.add_row({"Final gradient norm", format_double(grad_norm, 8)});
  verification_table.add_row({"||H^{-1}(H*v) - v|| error", format_double(error, 8)});
  
  std::cout << "Verification Results:\n" << verification_table << "\n";
  
  return 0;
}

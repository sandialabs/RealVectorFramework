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
#include <sstream>
#include <iomanip>
#include <tabulate/table.hpp>

#include "rvf.hpp"
#include "operations/std_cpo_impl.hpp"  // Enable std::vector support

using Vector = std::vector<double>;

// Helper function to configure table formatting consistently
void configure_table(tabulate::Table& table) {
  table.format()
    .border_top("─")
    .border_bottom("─") 
    .border_left("")
    .border_right("")
    .corner("")
    .column_separator("");
  
  // Add header separator after first row
  table[0].format().border_bottom("─");
}


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
    
    auto grad_cl = rvf::clone(x); auto& grad = deref_if_needed(grad_cl);
    auto step_cl = rvf::clone(x); auto& step = deref_if_needed(step_cl);  
    auto x_new_cl = rvf::clone(x); auto& x_new = deref_if_needed(x_new_cl);
    
    // Create TruncatedCG solver instance (once, reuse for efficiency)
    rvf::TruncatedCG<Objective, Vector> tcg_solver(obj, x);
    
    // Create iteration results table
    tabulate::Table iteration_table;
    iteration_table.add_row({"Iter", "f(x)", "||grad||", "delta", "TCG_flag", "TCG_iter", "rho"});
    iteration_table.format().font_style({tabulate::FontStyle::bold});
    
    // Configure table borders - only top, bottom, and header separator
    iteration_table.format()
      .border_top("─")
      .border_bottom("─") 
      .border_left("")
      .border_right("")
      .corner("")
      .column_separator("");
    
    // Add header separator after first row
    iteration_table[0].format()
      .border_bottom("─");
    
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
      
      // Set up Truncated CG parameters
      typename rvf::TruncatedCG<Objective, Vector>::Params tcg_params;
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
      
      // Add iteration info to table with scientific notation for better readability
      std::ostringstream f_x_str, grad_norm_str, delta_str, rho_str;
      f_x_str << std::scientific << std::setprecision(6) << f_x;
      grad_norm_str << std::scientific << std::setprecision(6) << grad_norm;
      delta_str << std::fixed << std::setprecision(6) << delta_;
      rho_str << std::fixed << std::setprecision(6) << rho;
      
      iteration_table.add_row({
        std::to_string(iter),
        f_x_str.str(),
        grad_norm_str.str(), 
        delta_str.str(),
        std::to_string(static_cast<int>(tcg_result.status)),
        std::to_string(tcg_result.iter),
        rho_str.str()
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
    
    // Configure table borders
    solution_table.format()
      .border_top("─")
      .border_bottom("─") 
      .border_left("")
      .border_right("")
      .corner("")
      .column_separator("");
    
    // Add header separator after first row
    solution_table[0].format()
      .border_bottom("─");
    
    for (size_t i = 0; i < x.size(); ++i) {
      std::ostringstream value_str;
      value_str << std::scientific << std::setprecision(8) << x[i];
      solution_table.add_row({"x[" + std::to_string(i) + "]", value_str.str()});
    }
    
    std::ostringstream final_f_str;
    final_f_str << std::scientific << std::setprecision(8) << obj.value(x);
    solution_table.add_row({"f(x) final", final_f_str.str()});
    
    std::cout << "Final Solution:\n" << solution_table << "\n";
  }
private:
  double delta_;
  double eta1_, eta2_;
  double gamma1_, gamma2_;
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
  
  // Configure table borders
  initial_table.format()
    .border_top("─")
    .border_bottom("─") 
    .border_left("")
    .border_right("")
    .corner("")
    .column_separator("");
  
  // Add header separator after first row
  initial_table[0].format()
    .border_bottom("─");
  
  initial_table.add_row({"Problem dimension", std::to_string(n)});
  
  std::ostringstream initial_point;
  initial_point << "(";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) initial_point << ", ";
    initial_point << std::fixed << std::setprecision(6) << x[i];
  }
  initial_point << ")";
  initial_table.add_row({"Initial point", initial_point.str()});
  
  std::ostringstream initial_f_str;
  initial_f_str << std::scientific << std::setprecision(6) << zakharov.value(x);
  initial_table.add_row({"Initial f(x)", initial_f_str.str()});
  
  std::cout << "Problem Setup:\n" << initial_table << "\n\n";
  
  // Solve using trust region method with RVF components
  trust_region_solver solver;
  solver.solve(zakharov, x, 50, 1e-6);
  
  // TCG flags reference table
  tabulate::Table flags_table;
  flags_table.add_row({"Flag", "Meaning"});
  flags_table.format().font_style({tabulate::FontStyle::bold});
  
  // Configure table borders
  flags_table.format()
    .border_top("─")
    .border_bottom("─") 
    .border_left("")
    .border_right("")
    .corner("")
    .column_separator("");
  
  // Add header separator after first row
  flags_table[0].format()
    .border_bottom("─");
  
  flags_table.add_row({"0", "Converged"});
  flags_table.add_row({"1", "Maximum iterations"});
  flags_table.add_row({"2", "Negative curvature encountered"});
  flags_table.add_row({"3", "Trust region boundary reached"});
  flags_table.add_row({"4", "Preconditioner failure"});
  
  std::cout << "Truncated CG Termination Flags:\n" << flags_table << "\n";
  
  // Verification using RVF operations
  auto grad_cl = rvf::clone(x); auto& grad = deref_if_needed(grad_cl);
  zakharov.gradient(grad, x);
  double grad_norm = std::sqrt(inner_product(grad, grad));
  
  // Test inverse Hessian functionality
  auto v_cl = rvf::clone(x);      auto& v = deref_if_needed(v_cl);
  auto hv_cl = rvf::clone(x);     auto& hv = deref_if_needed(hv_cl);
  auto inv_hv_cl = rvf::clone(x); auto& inv_hv = deref_if_needed(inv_hv_cl);
  auto error_cl = rvf::clone(x);  auto& error_vec = deref_if_needed(error_cl);
  
  // Use v = (1,1,1,1,1) as test vector
  for(size_t i = 0; i < v.size(); ++i) {
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
  
  // Configure table borders
  verification_table.format()
    .border_top("─")
    .border_bottom("─") 
    .border_left("")
    .border_right("")
    .corner("")
    .column_separator("");
  
  // Add header separator after first row
  verification_table[0].format()
    .border_bottom("─");
  
  std::ostringstream grad_norm_str, error_str;
  grad_norm_str << std::scientific << std::setprecision(8) << grad_norm;
  error_str << std::scientific << std::setprecision(8) << error;
  
  verification_table.add_row({"Final gradient norm", grad_norm_str.str()});
  verification_table.add_row({"||H^{-1}(H*v) - v|| error", error_str.str()});
  
  std::cout << "Verification Results:\n" << verification_table << "\n";
  
  return 0;
}

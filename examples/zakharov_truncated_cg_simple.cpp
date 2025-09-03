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

// Helper function for formatting doubles
std::string format_double(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

using Vector = std::vector<double>;

// Simple Zakharov function implementation without full RVF integration
class simple_zakharov {
private:
  Vector k_;
  double k_dot_k_;
  
  double dot(const Vector& a, const Vector& b) const {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      result += a[i] * b[i];
    }
    return result;
  }
  
public:
  explicit simple_zakharov(size_t n) : k_(n), k_dot_k_(0.0) {
    for (size_t i = 0; i < n; ++i) {
      k_[i] = static_cast<double>(i + 1);
      k_dot_k_ += k_[i] * k_[i];
    }
  }
  
  double value(const Vector& x) const {
    double x_dot_x = dot(x, x);
    double k_dot_x = dot(k_, x);
    return x_dot_x + 0.25 * k_dot_x * k_dot_x + 0.0625 * k_dot_x * k_dot_x * k_dot_x * k_dot_x;
  }
  
  void gradient(const Vector& x, Vector& grad) const {
    double k_dot_x = dot(k_, x);
    double coeff = 0.25 * (2.0 * k_dot_x + k_dot_x * k_dot_x * k_dot_x);
    
    for (size_t i = 0; i < x.size(); ++i) {
      grad[i] = 2.0 * x[i] + coeff * k_[i];
    }
  }
  
  void hess_vec(const Vector& x, const Vector& v, Vector& hv) const {
    double k_dot_x = dot(k_, x);
    double k_dot_v = dot(k_, v);
    double coeff = 0.25 * (2.0 + 3.0 * k_dot_x * k_dot_x) * k_dot_v;
    
    for (size_t i = 0; i < x.size(); ++i) {
      hv[i] = 2.0 * v[i] + coeff * k_[i];
    }
  }
};

// Simple truncated CG implementation
struct tcg_result {
  double step_norm = 0.0;
  double pred_reduction = 0.0;
  int flag = 0;
  int iterations = 0;
};

tcg_result truncated_cg_simple(
  const simple_zakharov& obj,
  const Vector& x,
  const Vector& grad,
  Vector& step,
  double delta,
  int max_iter = 20,
  double abs_tol = 1e-4,
  double rel_tol = 1e-2) {
  
  const double zero = 0.0, one = 1.0, two = 2.0, half = 0.5;
  size_t n = x.size();
  
  tcg_result result;
  
  // Initialize step to zero
  std::fill(step.begin(), step.end(), 0.0);
  
  // Working vectors
  Vector r = grad;  // residual = grad (since step = 0 initially)
  Vector p(n);    // search direction
  Vector Hp(n);   // H*p
  Vector s_tmp(n);  // temporary step
  
  double gnorm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));
  const double gtol = std::min(abs_tol, rel_tol * gnorm);
  
  // Initialize search direction p = -r
  for (size_t i = 0; i < n; ++i) {
    p[i] = -r[i];
  }
  
  double rho = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
  if (rho <= 0) {
    result.flag = 4;
    return result;
  }
  
  double snorm2 = 0.0;
  
  // Main CG iteration
  for (int iter = 0; iter < max_iter; ++iter) {
    result.iterations = iter;
    
    // Compute H*p
    obj.hessVec(Hp, p, x);
    
    // Check for negative curvature
    double kappa = std::inner_product(p.begin(), p.end(), Hp.begin(), 0.0);
    if (kappa <= 0) {
      // Find intersection with trust region boundary
      double sMp = std::inner_product(step.begin(), step.end(), p.begin(), 0.0);
      double pnorm2 = std::inner_product(p.begin(), p.end(), p.begin(), 0.0);
      double sigma = (-sMp + std::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
      
      // Update step
      for (size_t i = 0; i < n; ++i) {
        step[i] += sigma * p[i];
      }
      
      result.step_norm = delta;
      result.flag = 2; // Negative curvature
      result.pred_reduction += sigma * (rho - half * sigma * kappa);
      return result;
    }
    
    // Compute step length
    double alpha = rho / kappa;
    
    // Compute trial step
    for (size_t i = 0; i < n; ++i) {
      s_tmp[i] = step[i] + alpha * p[i];
    }
    
    // Check trust region constraint
    double s1norm2 = std::inner_product(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), 0.0);
    if (s1norm2 >= delta * delta) {
      // Find intersection with trust region boundary
      double sMp = std::inner_product(step.begin(), step.end(), p.begin(), 0.0);
      double pnorm2 = std::inner_product(p.begin(), p.end(), p.begin(), 0.0);
      double sigma = (-sMp + std::sqrt(sMp * sMp + pnorm2 * (delta * delta - snorm2))) / pnorm2;
      
      // Update step
      for (size_t i = 0; i < n; ++i) {
        step[i] += sigma * p[i];
      }
      
      result.step_norm = delta;
      result.flag = 3; // Trust region boundary
      result.pred_reduction += sigma * (rho - half * sigma * kappa);
      return result;
    }
    
    // Accept the step
    result.pred_reduction += half * alpha * rho;
    step = s_tmp;
    snorm2 = s1norm2;
    
    // Update residual
    for (size_t i = 0; i < n; ++i) {
      r[i] += alpha * Hp[i];
    }
    
    // Check convergence
    double normg = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));
    if (normg < gtol) {
      result.step_norm = std::sqrt(snorm2);
      result.flag = 0; // Converged
      result.iterations = iter + 1;
      return result;
    }
    
    // Update search direction
    double rho_new = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
    double beta = rho_new / rho;
    rho = rho_new;
    
    for (size_t i = 0; i < n; ++i) {
      p[i] = -r[i] + beta * p[i];
    }
  }
  
  // Maximum iterations reached
  result.step_norm = std::sqrt(snorm2);
  result.flag = 1; // Max iterations
  return result;
}

// Simple trust region solver
class simple_trust_region_solver {
private:
  double delta_;
  double eta1_, eta2_;
  double gamma1_, gamma2_;
  
public:
  simple_trust_region_solver(double delta_init = 1.0, double eta1 = 0.1, double eta2 = 0.75,
                double gamma1 = 0.5, double gamma2 = 2.0)
    : delta_(delta_init), eta1_(eta1), eta2_(eta2), gamma1_(gamma1), gamma2_(gamma2) {}
  
  void solve(simple_zakharov& obj, Vector& x, int max_iter = 100, double gtol = 1e-6) {
    Vector grad(x.size());
    Vector step(x.size());
    Vector x_new(x.size());
    
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
      double grad_norm = std::sqrt(std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
      
      // Check convergence
      if (grad_norm < gtol) {
        std::cout << "Converged with gradient norm: " << grad_norm << "\n";
        break;
      }
      
      // Solve trust region subproblem using Truncated CG
      std::fill(step.begin(), step.end(), 0.0);
      auto tcg_result = truncated_cg_simple(obj, x, grad, step, delta_);
      
      // Evaluate at trial point
      x_new = x;
      for (size_t i = 0; i < x.size(); ++i) {
        x_new[i] += step[i];
      }
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
        std::to_string(tcg_result.flag),
        std::to_string(tcg_result.iterations),
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
    
    // Print results
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
  const size_t n = 5;
  
  // Initial point
  Vector x(n, 3.0);
  
  // Create Zakharov objective
  simple_zakharov zakharov(n);
  
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
  
  // Solve using trust region method
  simple_trust_region_solver solver;
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
  
  // Final verification
  Vector grad(n);
  zakharov.gradient(grad, x);
  double grad_norm = std::sqrt(std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
  
  tabulate::Table verification_table;
  verification_table.add_row({"Test", "Result"});
  verification_table.format().font_style({tabulate::FontStyle::bold});
  verification_table.add_row({"Final gradient norm", format_double(grad_norm, 8)});
  
  std::cout << "Verification Results:\n" << verification_table << "\n";
  
  return 0;
}

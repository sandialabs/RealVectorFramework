/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "core/real_vector.hpp"
#include "operations/advanced/axpy_in_place.hpp"
#include "operations/core/add_in_place.hpp"
#include "operations/core/scale_in_place.hpp"
#include "operations/core/inner_product.hpp"
#include "operations/core/clone.hpp"
#include "operations/core/deref_if_needed.hpp"
#include "algorithms/optimization/line_search/gradient_descent_bounds.hpp" // For objective_function_c and linesearch
#include <vector>
#include <deque>

namespace rvf {

template<real_vector_c Vec>
class lbfgs_history {
public:
  using value_type = vector_value_t<Vec>;

  lbfgs_history(unsigned int m) : m_(m) {}

  void update(const Vec& s, const Vec& y) {
    if (s_list_.size() == m_) {
      s_list_.pop_front();
      y_list_.pop_front();
      rho_list_.pop_front();
    }
    s_list_.push_back(clone(s));
    y_list_.push_back(clone(y));
    rho_list_.push_back(value_type(1.0) / inner_product(s, y));
  }

  void apply_hessian_inverse(Vec& q) const {
    std::vector<value_type> alpha(m_);

    for (int i = s_list_.size() - 1; i >= 0; --i) {
      const auto& s_i_clone = s_list_[i];
      const auto& y_i_clone = y_list_[i];
      const auto& s_i = deref_if_needed(s_i_clone);
      const auto& y_i = deref_if_needed(y_i_clone);
      alpha[i] = rho_list_[i] * inner_product(s_i, q);
      axpy_in_place(q, -value_type(alpha[i]), y_i);
    }

    // This is a simplification. A proper implementation would allow
    // for different initial Hessian approximations. For now, we scale by a constant factor.
    if (!s_list_.empty()) {
      const auto& y_latest_clone = y_list_.back();
      const auto& s_latest_clone = s_list_.back();
      const auto& y_latest = deref_if_needed(y_latest_clone);
      const auto& s_latest = deref_if_needed(s_latest_clone);
      value_type gamma = inner_product(s_latest, y_latest) / inner_product(y_latest, y_latest);
      scale_in_place(q, gamma);
    }

    for (size_t i = 0; i < s_list_.size(); ++i) {
      const auto& y_i_clone = y_list_[i];
      const auto& s_i_clone = s_list_[i];
      const auto& y_i = deref_if_needed(y_i_clone);
      const auto& s_i = deref_if_needed(s_i_clone);
      value_type beta = rho_list_[i] * inner_product(y_i, q);
      axpy_in_place(q, value_type(alpha[i] - beta), s_i);
    }
  }

  void clear() {
    s_list_.clear();
    y_list_.clear();
    rho_list_.clear();
  }

private:
  unsigned int m_;
  std::deque<decltype(clone(std::declval<Vec>()))> s_list_;
  std::deque<decltype(clone(std::declval<Vec>()))> y_list_;
  std::deque<value_type> rho_list_;
};

template<typename Objective, real_vector_c Vec>
requires objective_function_c<Objective, Vec>
void lbfgs(
  const Objective& obj,
  Vec& x,
  unsigned int m = 5, // L-BFGS history size
  vector_value_t<Vec> gtol = 1e-6,
  vector_size_t<Vec> maxIter = 1000) {

  using value_type = vector_value_t<Vec>;
  
  linesearch_params<value_type> ls_params;
  lbfgs_history<Vec> history(m);

  auto x_old_cl = clone(x); auto& x_old = deref_if_needed(x_old_cl);
  auto grad_cl = clone(x);  auto& grad = deref_if_needed(grad_cl);
  auto grad_old_cl = clone(x); auto& grad_old = deref_if_needed(grad_old_cl);
  auto p_cl = clone(x); auto& p = deref_if_needed(p_cl); // Search direction
  auto s_cl = clone(x); auto& s = deref_if_needed(s_cl); // Step
  auto y_cl = clone(x); auto& y = deref_if_needed(y_cl); // Grad diff

  obj.gradient(grad, x);
  
  for(vector_size_t<Vec> iter = 0; iter < maxIter; ++iter) {
    if (rvf::sqrt(inner_product(grad, grad)) < gtol) {
      break;
    }

    // Compute search direction p = -H_k * g_k
    p = grad;
    scale_in_place(p, value_type(-1.0));
    history.apply_hessian_inverse(p);

    // Store old state
    x_old = x;
    grad_old = grad;

    // Perform line search
    // Note: This is a simplified line search for unconstrained problems.
    // A more robust implementation would use the one from gradient_descent_bounds
    // and handle bounds properly.
    value_type f_x = obj.value(x);
    const auto& const_p = p;
    value_type grad_dot_p = inner_product(grad, const_p);

    value_type alpha = ls_params.alpha_init;
    value_type f_trial;
    auto x_trial_cl = clone(x); auto& x_trial = deref_if_needed(x_trial_cl);

    for (int ls_iter = 0; ls_iter < ls_params.max_eval; ++ls_iter) {
      x_trial = x;
      axpy_in_place(x_trial, value_type(alpha), const_p);
      f_trial = obj.value(x_trial);
      if (f_trial <= f_x + ls_params.c1 * alpha * grad_dot_p) {
        break; // Armijo condition met
      }
      alpha *= ls_params.rho;
    }
    
    x = x_trial;

    // Update gradient
    obj.gradient(grad, x);

    // Update history
    s = x;
    const auto& const_x_old = x_old;
    axpy_in_place(s, value_type(-1.0), const_x_old);
    y = grad;
    const auto& const_grad_old = grad_old;
    axpy_in_place(y, value_type(-1.0), const_grad_old);

    history.update(s, y);
  }
}

} // namespace rvf

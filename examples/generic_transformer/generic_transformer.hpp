#pragma once

#include "core/rvf.hpp"
#include "core/type_support/std_cpo_impl.hpp"
#include "operations/advanced/self_map.hpp"
#include "operations/advanced/relu.hpp"
#include "operations/advanced/softmax.hpp"
#include "operations/advanced/layer_norm.hpp"
#include "operations/advanced/matvec.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include "operations/advanced/fill.hpp"
#include "operations/advanced/assign.hpp"
#include "operations/core/deref_if_needed.hpp"
#include "core/type_support/cmath/std_abs.hpp"
#include "core/type_support/cmath/std_exp.hpp"
#include "core/type_support/cmath/std_fmax.hpp"
#include "core/type_support/cmath/std_fmin.hpp"
#include "core/type_support/cmath/std_sqrt.hpp"
#include <vector>
#include <random>
#include <functional>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace transformer {

using namespace rvf;

template<rvf::real_vector_c V>
struct traits {
  using vector_type = V;
  using scalar_type = rvf::inner_product_return_t<V>;
  using matrix_type = std::vector<V>;
};

// ============================================================================
// COMPOSITION PRIMITIVES (vector-agnostic)
// ============================================================================

template<rvf::real_vector_c Vector>
using Matrix = typename traits<Vector>::matrix_type;

template<rvf::real_vector_c Vector>
auto make_linear_transform(const Matrix<Vector>& weights, const Vector& bias) {
  return [weights, bias](Vector& y, const Vector& x) {
    matvec(y, weights, x);
    add_in_place(y, bias);
  };
}

template<rvf::real_vector_c Vector>
auto make_activation(tincup::cpo_c auto activation_cpo) {
  return [activation_cpo](Vector& y, const Vector& x) {
    using Scalar = typename traits<Vector>::scalar_type;
    scale_in_place(y, Scalar(0));
    add_in_place(y, x);
    activation_cpo(y);
  };
}

template<rvf::real_vector_c Vector, typename Op1, typename Op2>
requires rvf::self_map_c<Op1, Vector> && rvf::self_map_c<Op2, Vector>
auto compose(Op1&& op1, Op2&& op2) {
  return [op1 = std::forward<Op1>(op1), op2 = std::forward<Op2>(op2)](Vector& y, const Vector& x) {
    auto temp_hold = rvf::clone(x);
    auto& temp = rvf::deref_if_needed(temp_hold);
    op1(temp, x);
    op2(y, temp);
  };
}

template<rvf::real_vector_c Vector, typename Op>
requires rvf::self_map_c<Op, Vector>
auto add_residual(Op&& op) {
  return [op = std::forward<Op>(op)](Vector& y, const Vector& x) {
    op(y, x);
    add_in_place(y, x);
  };
}

template<rvf::real_vector_c Vector, typename Op>
  requires rvf::self_map_c<Op, Vector>
auto add_layer_norm(Op&& op, typename traits<Vector>::scalar_type eps = static_cast<typename traits<Vector>::scalar_type>(1e-5)) {
  return [op = std::forward<Op>(op), eps](Vector& y, const Vector& x) {
    op(y, x);
    layer_norm(y, eps);
  };
}

// ============================================================================
// FEED-FORWARD NETWORK
// ============================================================================

template<rvf::real_vector_c Vector>
auto make_feed_forward(std::size_t d_model, std::size_t d_ff) {
  using Scalar = typename traits<Vector>::scalar_type;
  Matrix<Vector> W1(d_ff, Vector(d_model));
  Vector b1(d_ff, Scalar(0));
  Matrix<Vector> W2(d_model, Vector(d_ff));
  Vector b2(d_model, Scalar(0));

  std::random_device rd;
  std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix<Vector>& W, std::size_t in_dim, std::size_t out_dim) {
    Scalar std_dev = rvf::sqrt(Scalar(2) / Scalar(in_dim + out_dim));
    std::normal_distribution<std::remove_cvref_t<Scalar>> dist(Scalar(0), std_dev);
    auto randomize = [&](auto) { return dist(gen); };
    for (auto& row : W) unary_in_place(row, randomize);
  };

  init_matrix(W1, d_model, d_ff);
  init_matrix(W2, d_ff, d_model);

  return compose<Vector>(
    make_linear_transform<Vector>(W1, b1),
    compose<Vector>(
      make_activation<Vector>(relu),
      make_linear_transform<Vector>(W2, b2)
    )
  );
}

// ============================================================================
// SIMPLE ATTENTION (single-vector, pedagogical)
// ============================================================================

template<rvf::real_vector_c Vector>
auto make_attention(std::size_t d_model, std::size_t n_heads) {
  using Scalar = typename traits<Vector>::scalar_type;
  const std::size_t d_k = d_model / n_heads;

  Matrix<Vector> W_q(d_model, Vector(d_model));
  Matrix<Vector> W_k(d_model, Vector(d_model));
  Matrix<Vector> W_v(d_model, Vector(d_model));
  Matrix<Vector> W_o(d_model, Vector(d_model));

  std::random_device rd;
  std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix<Vector>& W, std::size_t in_dim, std::size_t out_dim) {
    Scalar std_dev = rvf::sqrt(Scalar(2) / Scalar(in_dim + out_dim));
    std::normal_distribution<std::remove_cvref_t<Scalar>> dist(Scalar(0), std_dev);
    auto randomize = [&](auto) { return dist(gen); };
    for (auto& row : W) unary_in_place(row, randomize);
  };

  init_matrix(W_q, d_model, d_model);
  init_matrix(W_k, d_model, d_model);
  init_matrix(W_v, d_model, d_model);
  init_matrix(W_o, d_model, d_model);

  return [W_q, W_k, W_v, W_o, d_k](Vector& y, const Vector& x) {
    auto Qh = rvf::clone(x); auto& Q = rvf::deref_if_needed(Qh);
    auto Kh = rvf::clone(x); auto& K = rvf::deref_if_needed(Kh);
    auto Vh = rvf::clone(x); auto& V = rvf::deref_if_needed(Vh);

    matvec(Q, W_q, x);
    matvec(K, W_k, x);
    matvec(V, W_v, x);

    using Scalar = typename traits<Vector>::scalar_type;
    Scalar attention_score = inner_product(Q, K) / rvf::sqrt(static_cast<Scalar>(d_k));

    auto Wh = rvf::clone(V); auto& weighted_V = rvf::deref_if_needed(Wh);
    scale_in_place(weighted_V, attention_score);
    matvec(y, W_o, weighted_V);
  };
}

// ============================================================================
// TRANSFORMER BLOCK AND MODEL
// ============================================================================

template<rvf::real_vector_c Vector>
auto make_transformer_block(std::size_t d_model, std::size_t n_heads, std::size_t d_ff) {
  auto attention = make_attention<Vector>(d_model, n_heads);
  auto feed_forward = make_feed_forward<Vector>(d_model, d_ff);
  return compose<Vector>(
    add_layer_norm<Vector>(add_residual<Vector>(attention)),
    add_layer_norm<Vector>(add_residual<Vector>(feed_forward))
  );
}

template<rvf::real_vector_c Vector>
auto make_transformer(std::size_t d_model, std::size_t n_layers, std::size_t n_heads, std::size_t d_ff) {
  std::function<void(Vector&, const Vector&)> transformer = [](Vector& y, const Vector& x) {
    using Scalar = typename traits<Vector>::scalar_type;
    scale_in_place(y, Scalar(0));
    add_in_place(y, x);
  };
  for (std::size_t i = 0; i < n_layers; ++i) {
    auto block = make_transformer_block<Vector>(d_model, n_heads, d_ff);
    transformer = compose<Vector>(std::move(transformer), std::move(block));
  }
  return transformer;
}

// ============================================================================
// GENERIC SEQUENTIAL / PARALLEL BUILDERS
// ============================================================================

template<rvf::real_vector_c Vector, typename... Ops>
requires (rvf::self_map_c<Ops, Vector> && ...)
auto make_sequential(Ops&&... ops) {
  return [ops...](Vector& y, const Vector& x) {
    auto curh = rvf::clone(x); auto& cur = rvf::deref_if_needed(curh);
    auto nxth = rvf::clone(x); auto& nxt = rvf::deref_if_needed(nxth);
    ((ops(nxt, cur), std::swap(cur, nxt)), ...);
    using Scalar = typename traits<Vector>::scalar_type;
    scale_in_place(y, Scalar(0));
    add_in_place(y, cur);
  };
}

template<rvf::real_vector_c Vector, typename... Ops>
requires (rvf::self_map_c<Ops, Vector> && ...)
auto make_parallel_sum(Ops&&... ops) {
  return [ops...](Vector& y, const Vector& x) {
    using Scalar = typename traits<Vector>::scalar_type;
    scale_in_place(y, Scalar(0));
    auto th = rvf::clone(x); auto& temp = rvf::deref_if_needed(th);
    ((ops(temp, x), add_in_place(y, temp)), ...);
  };
}

// ============================================================================
// SIMPLE TRAINING EXAMPLE (bias-only) — optional utility
// ============================================================================

template<rvf::real_vector_c Vector, typename Model>
requires rvf::self_map_c<Model, Vector>
struct GenericObjective {
  using Scalar = typename traits<Vector>::scalar_type;
  Model model;
  Vector input;
  Vector target;
  mutable Vector work;

  GenericObjective(Model m, const Vector& in, const Vector& t)
    : model(std::move(m)), input(in), target(t), work(rvf::clone(in)) {}

  Scalar value(const Vector& params) const {
    model(work, input);
    auto y = rvf::clone(work);
    // Build bias with target's size and copy common prefix from params
    auto bias = rvf::clone(target);
    rvf::fill(bias, Scalar(0));
    rvf::assign(bias, params);
    add_in_place(y, bias);
    auto diff = rvf::clone(y);
    auto neg_t = target;
    scale_in_place(neg_t, Scalar(-1));
    add_in_place(diff, neg_t);
    return Scalar(0.5) * inner_product(diff, diff);
  }

  void gradient(Vector& g, const Vector& params) const {
    model(work, input);
    auto y = rvf::clone(work);
    auto bias = params;
    add_in_place(y, bias);
    g = rvf::clone(params);
    rvf::fill(g, Scalar(0));
    auto diff = rvf::clone(y);
    auto neg_t = target;
    scale_in_place(neg_t, Scalar(-1));
    add_in_place(diff, neg_t);
    rvf::assign(g, diff);
  }
};

template<rvf::real_vector_c Vector, typename Model, typename Dataset>
requires rvf::self_map_c<Model, Vector>
void train_with_rvf(Model& model, const Dataset& dataset, std::size_t iterations = 100) {
  if (dataset.empty()) return;
  const auto& first = dataset.front();
  Vector params = rvf::clone(first.second);
  rvf::fill(params, typename traits<Vector>::scalar_type(0));
  Vector grad = params;

  for (std::size_t iter = 0; iter < iterations; ++iter) {
    rvf::fill(grad, typename traits<Vector>::scalar_type(0));
    for (const auto& [input, target] : dataset) {
      GenericObjective<Vector, Model> obj(model, input, target);
      Vector gsample = params;
      obj.gradient(gsample, params);
      add_in_place(grad, gsample);
    }
    scale_in_place(grad, typename traits<Vector>::scalar_type(-0.001));
    add_in_place(params, grad);
  }
}

} // namespace transformer
